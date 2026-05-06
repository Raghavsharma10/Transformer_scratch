def pairwise_extract(features, groups, edges, weight_func, use_orig_distr,
                     group_ids, num_groups, num_links,
                     non_symmetric, return_networkx_graph, out_weights_path):
    """
    Core function to compute the pair-wise histogram distance between all ROIs.

    Parameters
    ----------
    features
    groups
    edges
    weight_func
    use_orig_distr
    group_ids
    num_groups
    num_links
    non_symmetric
    return_networkx_graph
    out_weights_path

    Returns
    -------
    result : object
        A networkx graph or numpy array depending on request

    """

    # the following will execute only when the pair-wise computation is requested.
    if return_networkx_graph:
        graph = nx.DiGraph() if non_symmetric else nx.Graph()
        graph.add_nodes_from(group_ids)
    else:
        edge_weights = np.full([num_groups, num_groups], np.nan)

    exceptions_list = list()
    for src in range(num_groups):
        # primitive progress indicator
        if np.mod(src + 1, 5) == 0.0:
            sys.stdout.write('.')

        index1 = groups == group_ids[src]
        hist_one = compute_histogram(features[index1], edges, use_orig_distr)

        if non_symmetric:
            target_list = range(num_groups)
        else:
            # when symmetric, only upper tri matrix is computed/filled
            target_list = range(src + 1, num_groups, 1)

        for dest in target_list:
            # skipping edge between self
            if src == dest:
                continue

            index2 = groups == group_ids[dest]
            hist_two = compute_histogram(features[index2], edges, use_orig_distr)

            try:
                edge_value = weight_func(hist_one, hist_two)
                if return_networkx_graph:
                    graph.add_edge(group_ids[src], group_ids[dest], weight=float(edge_value))
                else:
                    edge_weights[src, dest] = edge_value
            except (RuntimeError, RuntimeWarning) as runexc:
                # placeholder to ignore some runtime errors (such as medpy's logger issue)
                print(runexc)
            except BaseException as exc:
                # numerical instabilities can cause trouble for histogram distance calculations
                traceback.print_exc()
                exceptions_list.append(str(exc))
                logging.warning('Unable to compute edge weight between '
                                ' {} and {}. Skipping it.'.format(group_ids[src], group_ids[dest]))

    error_thresh = 0.05
    if len(exceptions_list) >= error_thresh * num_links:
        print('All exceptions encountered so far:\n {}'.format('\n'.join(exceptions_list)))
        raise ValueError('Weights for atleast {:.2f}% of edges could not be computed.'.format(error_thresh * 100))

    sys.stdout.write('\n')

    if return_networkx_graph:
        if out_weights_path is not None:
            graph.write_graphml(out_weights_path)
        return graph
    else:
        if out_weights_path is not None:
            np.savetxt(out_weights_path, edge_weights, delimiter=',', fmt='%.9f')
        return edge_weights