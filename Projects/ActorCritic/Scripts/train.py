__author__ = "nasim.rahaman@iwr.uni-heidelberg.de"
__doc__ = """Train an actor-critic model."""

# Helper functions
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


def configure(actor, critic, modelconfig):
    """
    Configure actor and critic. This function should do the following:
        [+] set up loss variables for the actor and the critic
        [+] compute gradients
        [+] build the optimizer (with control variables)
        [+] compile training functions for both actor and critic
    """

    # actor and critic are not fedforward ICv1's. Actor takes x and outputs y, critic takes y and outputs l.
    # Feedforward actor
    actor.feedforward()
    # Feedforward critic with the output of the actor.
    critic.feedforward(inp=T.concatenate((actor.y, actor.x), axis=1))

    # Set up critic's loss
    # Make a relu function for the critic's loss
    relu = lambda x: T.switch(x > 0.)
    # Make k variable (for the critic). It has the shape (bs,)
    k = critic.baggage['k'] = T.vector('k')
    # Make loss. Note that critic.y.shape = (bs, 1, nr, nc). Convert to (bs, nr * nc) and sum along the second axis
    # before multiplying with k to save computation. The resulting vector of shape (bs,) (after having applied RELU) and
    # is averaged to obtain a scalar loss. The nash energy gives the loss at ground state.
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

    # Compile trainers
    # In addition to loss and cost, actor should also return it's output such that the critic can be trained.
    actor.classifiertrainer = A.function(inputs={'x': actor.x},
                                         outputs={'actor-C': actor.C, 'actor-L': actor.L, 'actor-y': actor.y},
                                         updates=actor.updates, allow_input_downcast=True, on_unused_input='warn')
    actor.classifier = A.function(inputs=[actor.x], outputs=actor.y, allow_input_downcast=True)

    # Remember that the input to the critic is the input and output to the actor, concatenated. However, the
    # concatenation happens in theano-space, so the compiled function takes just the input to and output from the actor
    # as its input.
    critic.classifiertrainer = A.function(inputs={'xx': actor.x, 'xy': actor.y, 'k': k},
                                          outputs={'critic-C': critic.C, 'critic-L': critic.L,
                                                   'critic-yc': critic.y.flatten(ndim=2).mean(axis=1), 'k': k},
                                          updates=critic.updates, allow_input_downcast=True, on_unused_input='warn')
    # Done.
    return actor, critic


def fit(actor, critic, trX, fitconfig, tools=None):
    """
    A customized training loop. This function should do the following:
        [-] train the actor and critic in tandem.
        [-] use a experience database to stabilize training the crtic
        [-] handle control variables
    """

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
                criticdatadeck.append({'x': batchX, 'y': batchY, 'k': np.array([1.])})
                actordatadeck.append({'x': batchX, 'y': batchY, 'k': np.array([1.])})
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

            if traincritic and iterstat['iternum'] % traincritic:
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

            if trainactor and iterstat['iternum']%trainactor:
                # Fetch batch for actor
                raw = fetch(actordatadeck)
                # Train actor
                actorout = actor.classifiertrainer(x=raw['x'])
                # Increment iteration counter
                iterstat['actor-iternum'] += 1
            else:
                # Skip training
                actorout = {}

            # Save actor
            if iterstat['actor-iternum'] % fitconfig['actor-save-every'] == 0:
                actor.save(nameflags='--iter-{}-routine'.format(iterstat['actor-iternum']))
                iterstat['actor-saved'] = True
            else:
                iterstat['actor-saved'] = False

            # Save critic
            if iterstat['critic-iternum'] % fitconfig['critic-save-every'] == 0:
                critic.save(nameflags='--iter-{}-routine'.format(iterstat['critic-iternum']))
                iterstat['critic-saved'] = True
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


def run(actor, critic, trX, runconfig):
    """
    Glue. This function's job:
        [-] set up callbacks.
        [-] set up experience database
        [-] configure actor and critic models (backup paths, etc.)
        [-] fit models within a try-except-finally clause
    """
    tools = {}
    # Set up relays
    if 'relayfile' in runconfig.keys():
        tools['relay'] = None
    # Set up callbacks
    pass
    # Set up logs
    pass
    # Configure model
    pass
    # Fit model
    pass


if __name__ == '__main__':
    print("[+] Initializing...")
    import argparse
    import yaml
    import sys
    import os
    import imp

    # Parse arguments
    parsey = argparse.ArgumentParser()
    parsey.add_argument("configset", help="Configuration file.")
    parsey.add_argument("--device", help="Device to use (overrides configuration file).", default=None)
    args = parsey.parse_args()

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

    from collections import deque
    import numpy as np
    import theano as th
    import theano.tensor as T

    # Add Antipasti to path
    sys.path.append('/export/home/nrahaman/Python/Antipasti')
    # Antipasti imports
    import Antipasti.netrain as nt
    import Antipasti.backend as A
    import Antipasti.trainkit as tk

    sys.path.append(os.path.normpath(os.path.join(__file__, "../../Boilerplate")))
    import tools