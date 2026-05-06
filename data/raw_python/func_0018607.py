def get_location(dom, location):
    """
    Get the node at the specified location in the dom.
    Location is a sequence of child indices, starting at the children of the
    root element. If there is no node at this location, raise a ValueError.
    """
    node = dom.documentElement
    for i in location:
        node = get_child(node, i)
        if not node:
            raise ValueError('Node at location %s does not exist.' % location) #TODO: line not covered
    return node