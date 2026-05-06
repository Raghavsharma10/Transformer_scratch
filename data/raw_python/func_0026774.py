def apply_repulsion(repulsion, nodes, barnes_hut_optimize=False, region=None, barnes_hut_theta=1.2):
    """
    Iterate through the nodes or edges and apply the forces directly to the node objects.
    """
    if not barnes_hut_optimize:
        for i in range(0, len(nodes)):
            for j in range(0, i):
                repulsion.apply_node_to_node(nodes[i], nodes[j])
    else:
        for i in range(0, len(nodes)):
            region.apply_force(nodes[i], repulsion, barnes_hut_theta)