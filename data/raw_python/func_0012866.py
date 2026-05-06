def dropnodes(edges):
    """draw a graph without the nodes"""
    newedges = []
    added = False
    for edge in edges:
        if bothnodes(edge):
            newtup = (edge[0][0], edge[1][0])
            newedges.append(newtup)
            added = True
        elif firstisnode(edge):
            for edge1 in edges:
                if edge[0] == edge1[1]:
                    newtup = (edge1[0], edge[1])
                    try:
                        newedges.index(newtup)
                    except ValueError as e:
                        newedges.append(newtup)
                    added = True
        elif secondisnode(edge):
            for edge1 in edges:
                if edge[1] == edge1[0]:
                    newtup = (edge[0], edge1[1])
                    try:
                        newedges.index(newtup)
                    except ValueError as e:
                        newedges.append(newtup)
                    added = True
        # gets the hanging nodes - nodes with no connection
        if not added:
            if firstisnode(edge):
                newedges.append((edge[0][0], edge[1]))
            if secondisnode(edge):
                newedges.append((edge[0], edge[1][0]))
        added = False
    return newedges