def parse_args(args):
    ''' Parse an argument string

        http://stackoverflow.com/questions/18160078/
        how-do-you-write-tests-for-the-argparse-portion-of-a-python-module
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', nargs='?',
                        help='Configuration yaml file', default=None)
    parser.add_argument(
        '--log', '-l',
        help='Logging level (e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL)',
        default='INFO')
    args_parsed = parser.parse_args(args)
    if not args_parsed.config_file:
        parser.error("You have to specify "
                     "a configuration file")  # pragma: no cover, sysexit
    return args_parsed