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

# ----------------------------------


def buildmodels(modelconfig):
    modelconfig = path2dict(modelconfig)
    # Import modelmaker
    modelmaker = imp.load_source('mmkr', modelconfig['path'])
    # Build models
    actor = modelmaker.build(**modelconfig['actor-buildconfig'])
    critic = modelmaker.build(**modelconfig['critic-buildconfig'])
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
    critic.feedforward(inp=T.concatenate((actor.y, actor.x), axis=1))

    # Set up critic's loss
    # Make a relu function for the critic's loss
    relu = lambda x: T.switch(x > 0., x, 0.)
    # Make k variable (for the critic). It has the shape (bs,)
    k = critic.baggage['k'] = T.vector('k')
    # Make loss. Note that critic.y.shape = (bs, 1, nr, nc). Convert to (bs, nr * nc) and sum along the second axis
    # before multiplying with k to save computation. The resulting vector of shape (bs,) (after having applied RELU)
    # and is averaged to obtain a scalar loss. The nash energy gives the loss at ground state.
    critic.L = relu(k * (critic.y.flatten(ndim=2).mean(axis=1) + np.float32(modelconfig['nashenergy']))).mean()
    # Add regularizer
    critic.C = critic.L + nt.lp(critic.params, regterms=[(2, 0.0005)])
    # Compute gradients
    critic.dC = T.grad(critic.C, wrt=critic.params)
    # Done.

    # Set up actor's loss
    # This is simply the mean of the critic's output. Backprop takes care of the rest.
    actor.L = critic.y.mean()
    actor.C = actor.L + nt.lp(actor.params, regterms=[(2, 0.0005)])
    # Compute gradients
    actor.dC = T.grad(actor.C, wrt=actor.params)
    # Done.

    # Set up optimizers
    if 'learningrate' in actor.baggage.keys():
        actor.getupdates(method='adam', learningrate=actor.baggage['learningrate'])
    else:
        actor.getupdates(method='adam')

    if 'learningrate' in critic.baggage.keys():
        critic.getupdates(method='adam', learningrate=critic.baggage['learningrate'])
    else:
        critic.getupdates(method='adam')

    print("[+] Compiling Actor...")
    # Compile trainers
    # In addition to loss and cost, actor should also return it's output such that the critic can be trained.
    actor.classifiertrainer = A.function(inputs={'x': actor.x},
                                         outputs={'actor-C': actor.C, 'actor-L': actor.L, 'actor-y': actor.y},
                                         updates=actor.updates, allow_input_downcast=True, on_unused_input='warn')
    actor.classifier = A.function(inputs=[actor.x], outputs=actor.y, allow_input_downcast=True)

    print("[+] Compiling Critic...")
    # Remember that the input to the critic is the input and output to the actor, concatenated. However, the
    # concatenation happens in theano-space, so the compiled function takes just the input to and output from the actor
    # as its input.
    critic.classifiertrainer = A.function(inputs=OrderedDict([('xx', actor.x), ('xy', actor.y), ('k', k)]),
                                          outputs={'critic-C': critic.C, 'critic-L': critic.L,
                                                   'critic-y': critic.y.flatten(ndim=2).mean(axis=1), 'k': k},
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
    criticdatadeck = deque()
    actordatadeck = deque()
    # Initialize an experience buffer to hold batches from the actor
    edb = deque(maxlen=fitconfig['edb-maxlen'])

    # Epoch loop
    while True:
        # Break if required
        if iterstat['epochnum'] >= fitconfig['numepochs']:
            break
        if iterstat['iternum'] >= fitconfig['maxiter']:
            break

        # Restart data generator
        trX.restartgenerator()

        # Primary loop
        while True:
            # Break if required
            if iterstat['iternum'] >= fitconfig['maxiter']:
                break

            # Get batch
            try:
                batchX, batchY = trX.next()
                # Append to criticdatadeck
                # Note: k is a vector of shape (bs,). While numpy broadcasting magic can handle
                # broadcasting a (1,) vector against (bs,), Theano can't.
                criticdatadeck.append({'x': batchX, 'y': batchY, 'k': np.zeros(shape=(batchX.shape[0],))})
                actordatadeck.append({'x': batchX, 'y': batchY, 'k': np.zeros(shape=(batchX.shape[0],))})
            except StopIteration:
                # Iterator might have stopped, but there could be batches left in the criticdatadeck
                if (len(criticdatadeck) + len(actordatadeck)) == 0:
                    # Ok, so the datadecks are out. Time to say goodbye.
                    break
                else:
                    # Still stuff left in the criticdatadeck, hurrah!
                    pass

            # Read relays
            if 'relay' in tools.keys():
                tools['relay']()

            # Read actor and critic training signals
            trainactor = tools['relay'].switches['actor-training-signal'].get_value()
            traincritic = tools['relay'].switches['critic-training-signal'].get_value()

            if traincritic and iterstat['iternum'] % traincritic == 0:
                # Try to fetch from experience database
                exp = fetch(edb)
                raw = fetch(criticdatadeck)
                # Consolidate to a common batch for the classifier
                critbatch = consolidatebatches(exp, raw)

                # Train
                criticout = critic.classifiertrainer(xx=critbatch['x'], xy=critbatch['y'], k=critbatch['k'])
                # Increment iteration counter
                iterstat['critic-iternum'] += 1
            else:
                # Skip training
                criticout = {}

            if trainactor and iterstat['iternum'] % trainactor == 0:
                # Fetch batch for actor
                raw = fetch(actordatadeck)
                # Train actor
                actorout = actor.classifiertrainer(x=raw['x'])
                # Increment iteration counter
                iterstat['actor-iternum'] += 1
                # Add to experience database for future replay
                edb.append({'x': raw['x'], 'y': actorout['actor-y'], 'k': np.ones(shape=(raw['x'].shape[0],))})
            else:
                # Skip training
                actorout = {}

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

    tools = {}
    # Set up relays
    if 'relayfile' in runconfig.keys():
        print("[+] Using relay file from {}.".format(runconfig['relayfile']))
        tools['relay'] = tk.relay(switches={'actor-training-signal': th.shared(value=np.float32(1)),
                                            'critic-training-signal': th.shared(value=np.float32(1)),
                                            'actor-learningrate': actor.baggage['learningrate'],
                                            'critic-learningrate': critic.baggage['learningrate']},
                                  ymlfile=runconfig['relayfile'])
    else:
        print("[-] Not listening to relays.")

    # Set up printer
    if 'verbose' in runconfig.keys() and runconfig['verbose']:
        tools['printer'] = tk.printer(monitors=[tk.monitorfactory('Actor-Cost', 'actor-C', float),
                                                tk.monitorfactory('Actor-Loss', 'actor-L', float),
                                                tk.monitorfactory('Critic-Cost', 'critic-C', float),
                                                tk.monitorfactory('Critic-Loss', 'critic-L', float),
                                                tk.monitorfactory('Critic-Prediction', 'critic-y', float)])

    # Set up logs
    if 'logfile' in runconfig.keys():
        print("[+] Logging to {}.".format(runconfig['logfile']))
        tools['log'] = tk.logger(runconfig['logfile'])
        # Bind logger to printer if possible
        if 'printer' in tools.keys():
            tools['printer'].textlogger = tools['log']
    else:
        print("[-] Not logging.")

    # Set up live plots
    if runconfig['live-plots']:
        print("[+] Live plots ON.")
        tools['plotter'] = tk.plotter(linenames=['actor-L', 'critic-L'], colors=['navy', 'firebrick'])
    else:
        print("[+] Live plots OFF.")

    # Gather all callbacks to a single object (tools != callbacks)
    callbacklist = []
    if 'printer' in tools.keys():
        callbacklist.append(tools['printer'])
    if 'plotter' in tools.keys():
        callbacklist.append(tools['plotter'])
    tools['callbacks'] = tk.callbacks(callbacklist)

    print("[+] Fitting...")
    # Fit models
    try:
        actor, critic = fit(actor, critic, trX, runconfig['fitconfig'], tools=tools)
    finally:
        actor.save(nameflags='-final')
        critic.save(nameflags='-final')

    # Return
    return actor, critic


if __name__ == '__main__':
    print("[+] Initializing...")
    import argparse
    import yaml
    import sys
    import imp
    import socket

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
        configset = '/home/nrahaman/LittleHeronHDD2/Neuro/ConvNet-Backups/ActorCritic/ICv1-ICv1-CREMI-0/' \
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

    # Go!
    run(config)
