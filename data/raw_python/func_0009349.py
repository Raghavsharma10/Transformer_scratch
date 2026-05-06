def is_component(w, ids):
    """Check if the set of ids form a single connected component

    Parameters
    ----------

    w   : spatial weights boject

    ids : list
          identifiers of units that are tested to be a single connected
          component


    Returns
    -------

    True    : if the list of ids represents a single connected component

    False   : if the list of ids forms more than a single connected component

    """

    components = 0
    marks = dict([(node, 0) for node in ids])
    q = []
    for node in ids:
        if marks[node] == 0:
            components += 1
            q.append(node)
            if components > 1:
                return False
        while q:
            node = q.pop()
            marks[node] = components
            others = [neighbor for neighbor in w.neighbors[node]
                      if neighbor in ids]
            for other in others:
                if marks[other] == 0 and other not in q:
                    q.append(other)
    return True