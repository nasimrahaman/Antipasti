__author__ = "nasim.rahaman@iwr.uni-heidelberg.de"
__doc__ = """Train an actor-critic model."""


# ---------Helper functions---------

def fetch(deck, rotate=False):
    """Fetch from a deque. Return None if nothing could be fetched."""
    try:
        if not rotate:
            return deck.pop()
        else:
            out = deck.pop()
            deck.appendleft(out)
            return out
    except IndexError:
        return None


def consolidatebatches(*records):
    """
    Given a list of dictionaries (records) as stored in experience buffer (edb) and data buffer (datadeck),
    concatenate them all to one consolidated record.
    """
    # Get rid of any Nones
    records = filter(None, records)
    assert len(records) != 0, "Given records contain only None's."

    # Get a list of keys in a given record
    keys = records[0].keys()
    # Get output and return
    out = {key: np.concatenate([record[key] for record in records], axis=0) for key in keys}
    return out


def path2dict(path):
    if isinstance(path, str):
        return tk.yaml2dict(path)
    elif isinstance(path, dict):
        return path
    else:
        raise NotImplementedError


def concshuffle(tensor1, tensor2):
    """Concatenate tensor1 and tensor2 along axis=1 and shuffle.
    This assumes that tensor1.shape[1] == tensor2.shape[1] dd== """
    # Make srng
    srng = T.shared_randomstreams.RandomStreams()
    # Concatenate
    cat = T.concatenate((tensor1, tensor2), axis=1)
    # Shuffle
    shuffed = cat[:, T.squeeze(srng.permutation(n=2, size=(1,))), :, :]
    # Return
    return shuffed


def broadcastk(k, batch):
    """Given an integer, broadcast k to a vector while replicating batchsize times."""
    batchsize = batch['x'].shape[0]
    return np.repeat(k, batchsize)


# ----------------------------------


def buildmodels(modelconfig):
    modelconfig = path2dict(modelconfig)
    # Check if model path is the same for both actor and critic:
    if 'actor-path' in modelconfig.keys() and 'critic-path' in modelconfig.keys():
        # Make actor and critic individually
        actormodelmaker = imp.load_source('ammkr', modelconfig['actor-path'])
        criticmodelmaker = imp.load_source('cmmkr', modelconfig['critic-path'])
        actor = actormodelmaker.build(**modelconfig['actor-buildconfig'])
        critic = criticmodelmaker.build(**modelconfig['critic-buildconfig'])
    elif 'path' in modelconfig.keys():
        # Import modelmaker
        modelmaker = imp.load_source('mmkr', modelconfig['path'])
        # Build models
        actor = modelmaker.build(**modelconfig['actor-buildconfig'])
        critic = modelmaker.build(**modelconfig['critic-buildconfig'])
    else:
        raise NotImplementedError

    # Return
    return actor, critic


def fetchfeeder(runconfig):
    runconfig = path2dict(runconfig)
    # Load datafeeders
    dataplate = imp.load_source('dataplate', runconfig['dataplatepath'])
    # Fetch feeder and return
    trX = dataplate.fetchfeeder(runconfig['dataconf'])
    return trX


def configure(modelconfig):
    """
    Configure actor and critic. This function should do the following:
        [+] set up loss variables for the actor and the critic
        [+] compute gradients
        [+] build the optimizer (with control variables)
        [+] compile training functions for both actor and critic
        [+] set backup paths
    """
    modelconfig = path2dict(modelconfig)

    # Build actor and critic models
    actor, critic = buildmodels(modelconfig)

    # actor and critic are not fedforward ICv1's. Actor takes x and outputs y, critic takes y and outputs l.
    # Feedforward actor
    actor.feedforward()
    # Feedforward critic with the output of the actor.
    critic.feedforward(inp=T.concatenate((concshuffle(actor.y, actor.yt), actor.x), axis=1))

    # Set up critic's loss
    # The critic predicts the absolute energy difference adE between actor.y and actor.yt. The critic's optimizer must
    # therefore try to maximize |critic.y| = adE.
    # Naively, that'd be it. But in reality, critic could shoot adE to infinity (for all inputs) before the actor gets
    # the chance to do shit. This could happen when the first layer in the critic has ~ 0 weights, but the last layer
    # simply blows up a constant bias. The gradient will be killed by the time it reaches the actor. Well, shit.

    # To regularize the critic, occasionally feed it with the same two inputs (actor.y = actor.yt) and
    # make sure the relative energy is still 0. This requires a k-variable that tells if the pairs being fed in are
    # actually the same image (i.e. if adE should be 0).
    # k = +1 ===> two images different (GT and Prediction)
    # k = -1 ===> two images same (GT and GT or Pred and Pred)
    k = critic.baggage['k'] = T.vector('k')
    # Compute critic loss
    adE = -k * T.abs_(critic.y).flatten(ndim=2).mean(axis=1)
    # Build loss. Remember, maximizing adE is minimizing -adE. The k-factor tells whether the
    critic.L = adE.mean()
    # Build cost
    criticC = critic.L + nt.lp(critic.params, regterms=[(2, 0.0005)])
    # Cut off cost if the energy is larger than a threshold ('adE-max', controllable from a control file). \
    # This is to prevent the critic from getting too far ahead (so that the actor can't catch up)
    # 1. Arrange transport for control variable
    adEmax = critic.baggage['adE-max'] = th.shared(np.float32(1.0))
    # 2. Kill cost (also the L2 term, in this case)
    critic.C = T.switch(adE.mean() < adEmax, criticC, 0.)
    # FIXME: Think what's wrong with this
    # critic.C = T.switch((adE < k * adEmax).mean() > 0.5, criticC, 0.)
    # Compute gradients
    critic.dC = T.grad(critic.C, wrt=critic.params)

    # Set up actor's loss
    # The actor's job is to generate samples actor.y for which the critic's absolute energy difference is minimized.
    # However, to prevent the actor getting too far ahead (so that the critic can't catch up), training is cancelled if
    # the critic's critique gets below a threshold. Additionally, if the critic is too far ahead, supervision is kicked
    # in.

    # Add control variable that weights the supervised term (wrt the critic)
    sw = actor.baggage['supervision-weight'] = th.shared(np.float32(0.5))
    # Add control variable that kicks in supervision when the critic's adE gets above it.
    ski = actor.baggage['supervision-kickin'] = th.shared(np.float32(0.3))
    # Add control variable that cuts off critic's critique if it's below a certain energy value (0.)
    adEmin = actor.baggage['adE-min'] = th.shared(np.float32(0.))

    # Get critic's unaudited critique (for the actor)
    crit = T.abs_(critic.y).mean()
    # Get a critique map to see what the critic is criticizing
    actor.baggage['critique-map'] = T.grad(crit, wrt=actor.y)

    # Audit critique. If the crit (= adE) is below a certain threshold (adE-min), don't train actor with it.
    # This has two functions. First, if adE is low, the critic can't provide insightful critique for the actor. Second,
    # if ski > adE-min > adE, the actor is not trained until the critic catches up.
    critique = T.switch(crit > adEmin, crit, 0.)

    # Get supervised loss
    supervisedloss = actor.baggage['supervised-loss'] = nt.bce(ist=actor.y, soll=actor.yt)

    # Build actor's loss
    actor.L = T.switch(critique > ski, (1. - sw) * critique + sw * supervisedloss, critique)
    # Get L2 for actor. Cut it off if adE is less than adE-min, to prevent the actor from training to only reduce L2
    # norm.
    al2 = T.switch(crit > adEmin, nt.lp(actor.params, regterms=[(2, 0.0005)]), 0.)
    # Build actor's cost
    actor.C = actor.L + al2
    # Compute gradients
    actor.dC = T.grad(actor.C, wrt=actor.params)

    # Set up optimizers
    if 'learningrate' in actor.baggage.keys():
        actor.getupdates(method=modelconfig.get('optimizer', 'adam'), learningrate=actor.baggage['learningrate'])
    else:
        actor.getupdates(method=modelconfig.get('optimizer', 'adam'))

    if 'learningrate' in critic.baggage.keys():
        critic.getupdates(method=modelconfig.get('optimizer', 'adam'), learningrate=critic.baggage['learningrate'])
    else:
        critic.getupdates(method=modelconfig.get('optimizer', 'adam'))

    print("[+] Compiling Actor...")
    # Compile trainers
    # In addition to loss and cost, actor should also return it's output such that the critic can be trained.
    actor.classifiertrainer = A.function(inputs=OrderedDict([('x', actor.x), ('yt', actor.yt)]),
                                         outputs={'actor-C': actor.C, 'actor-L': actor.L, 'actor-y': actor.y,
                                                  'actor-critique': critique,
                                                  'actor-critique-map': actor.baggage['critique-map']},
                                         updates=actor.updates, allow_input_downcast=True, on_unused_input='warn')
    actor.classifier = A.function(inputs=[actor.x], outputs=actor.y, allow_input_downcast=True)

    print("[+] Compiling Critic...")
    # Remember that the input to the critic is the input and output to the actor, concatenated. However, the
    # concatenation happens in theano-space, so the compiled function takes just the input to and output from the actor
    # as its input.
    critic.classifiertrainer = A.function(inputs=OrderedDict([('xx', actor.x), ('xy', actor.y), ('xyt', actor.yt),
                                                              ('k', k)]),
                                          outputs={'critic-C': critic.C, 'critic-L': critic.L, 'critic-k': k,
                                                   'critic-y': critic.y, 'critic-adE': adE},
                                          updates=critic.updates, allow_input_downcast=True, on_unused_input='warn')

    # Backup directories for actor and critic
    actor.savedir = modelconfig['actor-savedir']
    critic.savedir = modelconfig['critic-savedir']

    # Done.
    return actor, critic


def fit(actor, critic, trX, fitconfig, tools=None):
    """
    A customized training loop. This function should do the following:
        [-] train the actor and critic in tandem.
        [-] use a experience database to stabilize training the crtic
        [-] handle control variables
    """

    print("[+] Setting up training loop...")

    fitconfig = path2dict(fitconfig)

    # Defaults
    fitconfig['maxiter'] = np.inf if fitconfig['maxiter'] is None else fitconfig['maxiter']

    # Defaults
    if tools is None:
        tools = {}

    # Set up training variables
    iterstat = {'iternum': 0,
                'epochnum': 0,
                'actor-iternum': 0,
                'critic-iternum': 0,
                'actor-saved': False,
                'critic-saved': False}

    # Initialize a deque to hold batches (training might be skipped once in a while at unknown intervals)
    actordatadeck = deque()
    # Initialize an experience buffer to hold batches from the actor
    edb = deque(maxlen=fitconfig['edb-maxlen'])

    print("[+] Ready to train.")

    # Epoch loop
    while True:
        # Break if required
        if iterstat['epochnum'] >= fitconfig['numepochs']:
            break
        if iterstat['iternum'] >= fitconfig['maxiter']:
            break

        print("Epoch {} of {}:".format(iterstat['epochnum'], fitconfig['numepochs']))

        # Restart data generator
        trX.restartgenerator()

        # Primary loop
        while True:
            try:
                # Break if required
                if iterstat['iternum'] >= fitconfig['maxiter']:
                    break

                # Get batch
                try:
                    batchX, batchY = trX.next()
                    # Append to actor datadeck (remember, there is no critic datadeck)
                    actordatadeck.append({'x': batchX, 'yt': batchY})
                except StopIteration:
                    # Iterator might have stopped, but there could be batches left in the actordatadeck
                    if len(actordatadeck) == 0:
                        break
                    else:
                        pass

                # Read relays
                if 'relay' in tools.keys():
                    tools['relay']()

                # Read actor and critic training signals
                trainactor = tools['relay'].switches['actor-training-signal'].get_value()
                traincritic = tools['relay'].switches['critic-training-signal'].get_value()

                if trainactor and iterstat['iternum'] % trainactor == 0 and len(actordatadeck) != 0:
                    # Fetch batch for actor
                    raw = fetch(actordatadeck)

                    # Train actor with GT
                    actorout = actor.classifiertrainer(x=raw['x'], yt=raw['yt'])
                    actorout.update({'actor-x': raw['x'], 'actor-yt': raw['yt']})

                    # Increment iteration counter
                    iterstat['actor-iternum'] += 1
                    # Add to experience database for future replay
                    edb.append({'x': raw['x'], 'y': actorout['actor-y'], 'yt': raw['yt']})
                else:
                    # Skip training
                    actorout = {}

                if traincritic and iterstat['iternum'] % traincritic == 0:
                    # Try to fetch from experience database
                    critbatch = fetch(edb)
                    # Choose k
                    k = np.random.choice([1, -1])
                    # Based on k, choose critbatch keys
                    if k == 1:
                        # y and yt are different.
                        xykey = 'y'
                        xytkey = 'yt'
                    else:
                        # y and yt are the same (either y or yt)
                        xytkey = xykey = np.random.choice(['y', 'yt'])

                    # Train
                    criticout = critic.classifiertrainer(xx=critbatch['x'], xy=critbatch[xykey], xyt=critbatch[xytkey],
                                                         k=broadcastk(k, critbatch))
                    criticout.update({'critic-xx': critbatch['x'], 'critic-xy': critbatch['y'],
                                      'critic-xyt': critbatch['yt']})

                    # Increment iteration counter
                    iterstat['critic-iternum'] += 1
                else:
                    # Skip training
                    criticout = {}

                # Save actor
                if iterstat['actor-iternum'] % fitconfig['actor-save-every'] == 0:
                    actor.save(nameflags='--iter-{}-routine'.format(iterstat['actor-iternum']))
                    iterstat['actor-saved'] = True
                    print("[+] Saved actor parameters to {}.".format(actor.lastsavelocation))
                else:
                    iterstat['actor-saved'] = False

                # Save critic
                if iterstat['critic-iternum'] % fitconfig['critic-save-every'] == 0:
                    critic.save(nameflags='--iter-{}-routine'.format(iterstat['critic-iternum']))
                    iterstat['critic-saved'] = True
                    print("[+] Saved critic parameters to {}.".format(critic.lastsavelocation))
                else:
                    iterstat['critic-saved'] = False

                # Update iterstat with actor and critic outputs
                iterstat.update(criticout)
                iterstat.update(actorout)

                # Callbacks
                if 'callbacks' in tools.keys():
                    tools['callbacks'](**iterstat)

                # Increment global iteration counter
                iterstat['iternum'] += 1

            except KeyboardInterrupt:
                print("\n[o] Starting Interface...")
                print("-------------------------")

                # Get interface input
                userinp = 'd'
                while True:
                    userinp = raw_input("\nb > break loop\nc > continue\nd > enter debugger (pdb)\nq > quit\n")
                    if userinp in ['b', 'c', 'd', 'q']:
                        break
                    else:
                        print("Invalid command.\n")
                        continue

                if userinp == 'b':
                    break

                elif userinp == 'c':
                    continue

                elif userinp == 'd':
                    pdb.set_trace()

                elif userinp == 'q':
                    raise KeyboardInterrupt

        # Increment epoch counter
        iterstat['epochnum'] += 1

    # Return trained actor and critic
    return actor, critic


def run(runconfig):
    """
    Glue. This function's job:
        [+] build and configure actor and critic models
        [+] set up data feeder
        [+] set up callbacks.
        [+] fit models within a try-except-finally clause
    """

    runconfig = path2dict(runconfig)

    print("[+] Building and configuring models...")
    # Configure model
    actor, critic = configure(runconfig['modelconfig'])
    print("[+] Loading data feeders...")
    # Load feeder
    trX = fetchfeeder(runconfig)

    print("[+] Setting up utilities...")
    # setupconfig is not just a copy of runconfig (!)
    setupconfig = runconfig
    setupconfig.update({'actor-learningrate': actor.baggage['learningrate'],
                        'critic-learningrate': critic.baggage['learningrate'],
                        'supervision-weight': actor.baggage['supervision-weight'],
                        'supervision-kickin': actor.baggage['supervision-kickin'],
                        'adE-max': critic.baggage['adE-max'],
                        'adE-min': actor.baggage['adE-min']})
    tools = setuptools(setupconfig)

    print("[+] Ready.")
    # Fit models
    try:
        actor, critic = fit(actor, critic, trX, runconfig['fitconfig'], tools=tools)
    finally:
        actor.save(nameflags='-final')
        critic.save(nameflags='-final')

    # Return
    return actor, critic


def setuptools(setupconfig):
    tools = {}
    # Set up relays
    if 'relayfile' in setupconfig.keys():
        print("[+] Using relay file from {}.".format(setupconfig['relayfile']))
        tools['relay'] = tk.relay(switches={'actor-training-signal': th.shared(value=np.float32(1)),
                                            'critic-training-signal': th.shared(value=np.float32(1)),
                                            'actor-learningrate': setupconfig['actor-learningrate'],
                                            'critic-learningrate': setupconfig['critic-learningrate'],
                                            'supervision-weight': setupconfig['supervision-weight'],
                                            'supervision-kickin': setupconfig['supervision-kickin'],
                                            'adE-max': setupconfig['adE-max'],
                                            'adE-min': setupconfig['adE-min']},
                                  ymlfile=setupconfig['relayfile'])
    else:
        print("[-] Not listening to relays.")

    # Set up printer
    if 'verbose' in setupconfig.keys() and setupconfig['verbose']:
        tools['printer'] = tk.printer(monitors=[tk.monitorfactory('Iteration', 'iternum', int),
                                                tk.monitorfactory('Actor-Cost', 'actor-C', float),
                                                tk.monitorfactory('Actor-Loss', 'actor-L', float),
                                                tk.monitorfactory('Actor-Critique', 'actor-critique', float),
                                                tk.monitorfactory('Critic-Cost', 'critic-C', float),
                                                tk.monitorfactory('Critic-Loss', 'critic-L', float),
                                                tk.monitorfactory('Critic-Performance', 'critic-performance', float),
                                                tk.monitorfactory('Critic-k', 'critic-k', float)])

    # Set up logs
    if 'logfile' in setupconfig.keys():
        print("[+] Logging to {}.".format(setupconfig['logfile']))
        tools['log'] = tk.logger(setupconfig['logfile'])
        # Bind logger to printer if possible
        if 'printer' in tools.keys():
            tools['printer'].textlogger = tools['log']
    else:
        print("[-] Not logging.")

    # Set up live plots
    if setupconfig['live-plots']:
        print("[+] Live plots ON.")
        tools['plotter'] = tk.plotter(linenames=['actor-L', 'critic-L'], colors=['navy', 'firebrick'])
    else:
        print("[+] Live plots OFF.")

    if setupconfig.get('live-print') is not None:

        print("[+] Network outputs will be printed "
              "to {} every {} iteratons.".format(setupconfig['live-print']['printdir'],
                                                 setupconfig['live-print']['every']))

        def outputprinter(**iterstat):
            if iterstat['iternum'] % setupconfig['live-print']['every'] == 0:

                if 'actor-y' in iterstat.keys():
                    # Print
                    vz.printensor2file(iterstat['actor-y'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='AY--'.format(iterstat['iternum']))

                if 'actor-x' in iterstat.keys():
                    vz.printensor2file(iterstat['actor-x'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='AX--'.format(iterstat['iternum']))

                if 'actor-yt' in iterstat.keys():
                    vz.printensor2file(iterstat['actor-yt'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='AYT--'.format(iterstat['iternum']))

                if 'actor-critique-map' in iterstat.keys():
                    vz.printensor2file(iterstat['actor-critique-map'], savedir=setupconfig['live-print']['printdir'],
                                       mode='image', nameprefix='ACM--'.format(iterstat['iternum']))

                if 'critic-y' in iterstat.keys() and iterstat['critic-y'].shape[2:] != (1, 1):
                    vz.printensor2file(iterstat['critic-y'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='CY--'.format(iterstat['iternum']))

                if 'critic-xx' in iterstat.keys():
                    vz.printensor2file(iterstat['critic-xx'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='CXX--'.format(iterstat['iternum']))

                if 'critic-xy' in iterstat.keys():
                    vz.printensor2file(iterstat['critic-xy'], savedir=setupconfig['live-print']['printdir'], mode='image',
                                       nameprefix='CXY--'.format(iterstat['iternum']))
            else:
                return

        tools['live-printer'] = tk.caller(outputprinter)

    # Gather all callbacks to a single object (tools != callbacks)
    callbacklist = []
    if 'printer' in tools.keys():
        callbacklist.append(tools['printer'])
    if 'plotter' in tools.keys():
        callbacklist.append(tools['plotter'])
    if 'live-printer' in tools.keys():
        callbacklist.append(tools['live-printer'])
    tools['callbacks'] = tk.callbacks(callbacklist)

    return tools


if __name__ == '__main__':
    print("[+] Initializing...")
    import argparse
    import yaml
    import sys
    import imp
    import socket
    import pdb

    # fatchicken runs Pycharm while everyone else relies on the CLI.
    isfatchicken = socket.gethostname() == 'fatchicken'

    # Parse arguments
    parsey = argparse.ArgumentParser()
    parsey.add_argument("configset", help="Configuration file.")
    parsey.add_argument("--device", help="Device to use (overrides configuration file).", default=None)

    # Pycharm debugging
    if not isfatchicken:
        args = parsey.parse_args()
    else:
        # Set up the interpreter to use the right Antipasti
        sys.path.append('/export/home/nrahaman/Python/Repositories/Antipasti/')
        # Hard-coded parameters
        configset = '/mnt/localdata02/nrahaman/Neuro/ConvNet-Backups/ActorCritic/ResNIN-ResNIN+-REGAN-CREMI-0/' \
                    'Configurations/simplerunconfig.yml'
        device = 'gpu0'
        args = argparse.Namespace(configset=configset, device=device)

    # Load configuration dict
    with open(args.configset) as configfile:
        config = yaml.load(configfile)

    print("[+] Using configuration file from {}.".format(args.configset))

    # Read which device to use
    if args.device is None:
        device = config['device']
    else:
        device = args.device

    assert device is not None

    print("[+] Using device {}.".format(device))

    # Import shit
    from theano.sandbox.cuda import use
    use(device)

    from collections import deque, OrderedDict
    import numpy as np
    import theano as th
    import theano.tensor as T

    # Add Antipasti to path
    sys.path.append('/export/home/nrahaman/Python/Antipasti')
    # Antipasti imports
    import Antipasti.netrain as nt
    import Antipasti.backend as A
    import Antipasti.trainkit as tk
    import Antipasti.vizkit as vz

    # Go!
    run(config)
