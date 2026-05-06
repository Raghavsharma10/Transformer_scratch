def cli():
    """ Allow the module to be called from the cli. """
    import argparse

    parser = argparse.ArgumentParser(description='Send data to graphite')

    # Core of the application is to accept a metric and a value.
    parser.add_argument('metric', metavar='metric', type=str,
                        help='name.of.metric')
    parser.add_argument('value', metavar='value', type=int,
                        help='value of metric as int')

    args = parser.parse_args()
    metric = args.metric
    value = args.value

    graphitesend_instance = init()
    graphitesend_instance.send(metric, value)