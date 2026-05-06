def greedy_trails(subg, odds, verbose):
    '''Greedily select trails by making the longest you can until the end'''
    if verbose:
        print('\tCreating edge map')

    edges = defaultdict(list)

    for x,y in subg.edges():
        edges[x].append(y)
        edges[y].append(x)

    if verbose:
        print('\tSelecting trails')

    trails = []
    for x in subg.nodes():
        if verbose > 2:
            print('\t\tNode {0}'.format(x))

        while len(edges[x]) > 0:
            y = edges[x][0]
            trail = [(x,y)]
            edges[x].remove(y)
            edges[y].remove(x)
            while len(edges[y]) > 0:
                x = y
                y = edges[y][0]
                trail.append((x,y))
                edges[x].remove(y)
                edges[y].remove(x)
            trails.append(trail)
    return trails