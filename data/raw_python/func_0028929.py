def _add_default_arguments(parser):
    """Add the default arguments to the parser.

    :param argparse.ArgumentParser parser: The argument parser

    """
    parser.add_argument('-c', '--config', action='store', dest='config',
                        help='Path to the configuration file')
    parser.add_argument('-f', '--foreground', action='store_true', dest='foreground',
                        help='Run the application interactively')