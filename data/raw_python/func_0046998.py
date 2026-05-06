def get_parser():
    "Specifies the arguments and defaults, and returns the parser."

    parser = argparse.ArgumentParser(prog="hiwenet")

    parser.add_argument("-f", "--in_features_path", action="store", dest="in_features_path",
                        required=True,
                        help="Abs. path to file containing features for a given subject")

    parser.add_argument("-g", "--groups_path", action="store", dest="groups_path",
                        required=True,
                        help="path to a file containing element-wise membership into groups/nodes/patches.")

    parser.add_argument("-w", "--weight_method", action="store", dest="weight_method",
                        default=default_weight_method, required=False,
                        help="Method used to estimate the weight between the pair of nodes. Default : {}".format(
                            default_weight_method))

    parser.add_argument("-o", "--out_weights_path", action="store", dest="out_weights_path",
                        default=default_out_weights_path, required=False,
                        help="Where to save the extracted weight matrix. If networkx output is returned, it would be saved in GraphML format. Default: nothing saved.")

    parser.add_argument("-n", "--num_bins", action="store", dest="num_bins",
                        default=default_num_bins, required=False,
                        help="Number of bins used to construct the histogram. Default : {}".format(default_num_bins))

    parser.add_argument("-r", "--edge_range", action="store", dest="edge_range",
                        default=default_edge_range, required=False,
                        nargs = 2,
                        help="The range of edges (two finite values) within which to bin the given values e.g. --edge_range 1 6 "
                             "This can be helpful to ensure correspondence across multiple invocations of hiwenet (for different subjects),"
                             " in terms of range across all bins as well as individual bin edges. "
                             "Default : {}, to automatically compute from the given values.".format(default_edge_range))

    parser.add_argument("-t", "--trim_outliers", action="store", dest="trim_outliers",
                        default=default_trim_behaviour, required=False,
                        help="Boolean flag indicating whether to trim the extreme/outlying values. Default True.")

    parser.add_argument("-p", "--trim_percentile", action="store", dest="trim_percentile",
                        default=default_trim_percentile, required=False,
                        help="Small value specifying the percentile of outliers to trim. "
                             "Default: {0}%% , must be in open interval (0, 100).".format(default_trim_percentile))

    parser.add_argument("-x", "--return_networkx_graph", action="store", dest="return_networkx_graph",
                        default=default_return_networkx_graph, required=False,
                        help="Boolean flag indicating whether to return a networkx graph populated with weights computed. Default: False")

    return parser