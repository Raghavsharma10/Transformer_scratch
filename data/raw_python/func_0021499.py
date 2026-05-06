def decompose_graph(g, heuristic='tour', max_odds=20, verbose=0):
    '''Decompose a graph into a set of non-overlapping trails.'''
    # Get the connected subgraphs
    subgraphs = [nx.subgraph(g, x).copy() for x in nx.connected_components(g)]

    chains = []
    num_subgraphs = len(subgraphs)
    step = 0
    while num_subgraphs > 0:
        if verbose:
            print('Step #{0} ({1} subgraphs)'.format(step, num_subgraphs))

        for i in range(num_subgraphs-1, -1, -1):
            subg = subgraphs[i]

            # Get all odd-degree nodes
            odds = [x for x,y in dict(nx.degree(subg)).items() if y % 2 == 1]

            if verbose > 1:
                if len(odds) == 0:
                    print('\t\tNo odds')
                elif len(odds) == 2:
                    print('\t\tExactly 2 odds')
                else:
                    print('\t\t{0} odds'.format(len(odds)))
            
            # If there are no odd-degree edges, we can find an euler circuit
            if len(odds) == 0:
                trails = [list(nx.eulerian_circuit(subg))]
            elif len(odds) == 2:
                # If there are only two odd-degree edges, we can find an euler tour
                trails = [calc_euler_tour(subg, odds[0], odds[1])]
            elif heuristic in ['min', 'max', 'median', 'any']:
                trails = select_odd_degree_trail(subg, odds, max_odds, heuristic, verbose)
            elif heuristic == 'random':
                trails = select_random_trail(subg, verbose)
            elif heuristic == 'mindegree':
                trails = select_min_degree_trail(subg, max_odds, verbose)
            elif heuristic == 'ones':
                trails = select_single_edge_trails(subg, verbose)
            elif heuristic == 'tour':
                trails = pseudo_tour_trails(subg, odds, verbose)
            elif heuristic == 'greedy':
                trails = greedy_trails(subg, odds, verbose)

            if verbose > 2:
                print('\t\tTrails: {0}'.format(len(trails)))

            # Remove the trail
            for trail in trails:
                subg.remove_edges_from(trail)

            # Add it to the list of chains
            chains.extend(trails)
            
            # If the subgraph is empty, remove it from the list
            if subg.number_of_edges() == 0:
                del subgraphs[i]
            else:
                comps = list(nx.connected_components(subg))

                # If the last edge split the graph, add the new subgraphs to the list of subgraphs
                if len(comps) > 1:
                    for x in comps:
                        compg = nx.subgraph(subg, x)
                        if compg.number_of_edges() > 0:
                            subgraphs.append(compg)
                    del subgraphs[i]

        # Update the count of connected subgraphs
        num_subgraphs = len(subgraphs)
        step += 1

    return chains