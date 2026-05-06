def parse_args():
    """Parser/validator for the cmd line args."""

    parser = get_parser()

    if len(sys.argv) < 2:
        parser.print_help()
        warnings.warn('Too few arguments!', UserWarning)
        parser.exit(1)

    # parsing
    try:
        params = parser.parse_args()
    except Exception as exc:
        print(exc)
        raise ValueError('Unable to parse command-line arguments.')

    in_features_path = os.path.abspath(params.in_features_path)
    if not os.path.exists(in_features_path):
        raise IOError("Given features file doesn't exist.")

    groups_path = os.path.abspath(params.groups_path)
    if not os.path.exists(groups_path):
        raise IOError("Given groups file doesn't exist.")

    return in_features_path, groups_path, params.weight_method, params.num_bins, params.edge_range, \
           params.trim_outliers, params.trim_percentile, params.return_networkx_graph, params.out_weights_path