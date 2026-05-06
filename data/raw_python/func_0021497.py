def calc_euler_tour(g, start, end):
    '''Calculates an Euler tour over the graph g from vertex start to vertex end.
    Assumes start and end are odd-degree vertices and that there are no other odd-degree
    vertices.'''
    even_g = nx.subgraph(g, g.nodes()).copy()
    if end in even_g.neighbors(start):
        # If start and end are neighbors, remove the edge
        even_g.remove_edge(start, end)
        comps = list(nx.connected_components(even_g))
        # If the graph did not split, just find the euler circuit
        if len(comps) == 1:
            trail = list(nx.eulerian_circuit(even_g, start))
            trail.append((start, end))
        elif len(comps) == 2:
            subg1 = nx.subgraph(even_g, comps[0])
            subg2 = nx.subgraph(even_g, comps[1])
            start_subg, end_subg = (subg1, subg2) if start in subg1.nodes() else (subg2, subg1)
            trail = list(nx.eulerian_circuit(start_subg, start)) + [(start, end)] + list(nx.eulerian_circuit(end_subg, end))
        else:
            raise Exception('Unknown edge case with connected components of size {0}:\n{1}'.format(len(comps), comps))
    else:
        # If they are not neighbors, we add an imaginary edge and calculate the euler circuit
        even_g.add_edge(start, end)
        circ = list(nx.eulerian_circuit(even_g, start))
        try:
            trail_start = circ.index((start, end))
        except:
            trail_start = circ.index((end, start))
        trail = circ[trail_start+1:] + circ[:trail_start]
    return trail