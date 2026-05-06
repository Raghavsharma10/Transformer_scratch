def add_node(node, parent):
    '''add_node will add a node to it's parent
    '''
    newNode = dict(node_id=node.id, children=[])
    parent["children"].append(newNode)
    if node.left: add_node(node.left, newNode)
    if node.right: add_node(node.right, newNode)