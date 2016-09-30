__doc__ = """General script for running networks."""



def fetchfromdict(dictionary, key, default=None):
    if key in dictionary.keys():
        return dictionary[key]
    else:
        return default


def plot(net, trX, **plotconfig):
    # Aliases
    ffd = fetchfromdict
    # TODO
    pass


if __name__ == '__main__':
    import argparse
    import yaml
    import sys
    import imp

    # Parse arguments
    parsey = argparse.ArgumentParser()
    parsey.add_argument("configset", help="Configuration file.")
    parsey.add_argument("--device", help="Device to use (overrides configuration file).", default=None)
    args = parsey.parse_args()

    # Load configuration dict
    with open(args.configset) as configfile:
        config = yaml.load(configfile)

    # Read which device to use
    if args.device is None:
        device = config['device']
    else:
        device = args.device

    assert device is not None

    # Import shit
    from theano.sandbox.cuda import use
    use(device)

    # Add Antipasti to path
    sys.path.append('/export/home/nrahaman/Python/Antipasti')
    import Antipasti.trainkit as tk
    import Antipasti.netutils as nu

    # Import model
    model = imp.load_source('model', config['modelpath'])
    # Import datafeeder
    dpl = imp.load_source('dataplate', config['dplpath'])

    # Build network
    net = model.build(**config['buildconfig'])

    # Fetch datafeeders
    trX = dpl.fetchfeeder(config['dataconf'])

    # Run training
    try:
        plot(net, trX, **config['plotconfig'])
    except Exception as e:
        raise e
    finally:
        trX.cleanup()
