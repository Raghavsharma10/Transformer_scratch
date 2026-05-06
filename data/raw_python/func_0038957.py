def get_parser():
    """Argument specifier.

    """

    parser = argparse.ArgumentParser(prog='pyradigm')

    parser.add_argument('path_list', nargs='*', action='store',
                        default=None, help='List of paths to display info about.')

    parser.add_argument('-m', '--meta', action='store_true', dest='meta_requested',
                        required=False,
                        default=False, help='Prints the meta data (subject_id,class).')

    parser.add_argument('-i', '--info', action='store_true', dest='summary_requested',
                        required=False,
                        default=False,
                        help='Prints summary info (classes, #samples, #features).')

    arithmetic_group = parser.add_argument_group('Options for multiple datasets')
    arithmetic_group.add_argument('-a', '--add', nargs='+', action='store',
                                  dest='add_path_list', required=False,
                                  default=None,
                                  help='List of MLDatasets to combine')

    arithmetic_group.add_argument('-o', '--out_path', action='store', dest='out_path',
                                  required=False,
                                  default=None,
                                  help='Output path to save the resulting dataset.')

    return parser