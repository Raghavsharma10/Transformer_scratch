def run_cli():
    "Command line interface to hiwenet."

    features_path, groups_path, weight_method, num_bins, edge_range, \
    trim_outliers, trim_percentile, return_networkx_graph, out_weights_path = parse_args()

    # TODO add the possibility to process multiple combinations of parameters: diff subjects, diff metrics
    # for features_path to be a file containing multiple subjects (one/line)
    # -w could take multiple values kldiv,histint,
    # each line: input_features_path,out_weights_path

    features, groups = read_features_and_groups(features_path, groups_path)

    extract(features, groups, weight_method=weight_method, num_bins=num_bins,
            edge_range=edge_range, trim_outliers=trim_outliers, trim_percentile=trim_percentile,
            return_networkx_graph=return_networkx_graph, out_weights_path=out_weights_path)