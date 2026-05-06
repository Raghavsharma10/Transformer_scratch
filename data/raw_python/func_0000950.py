def body_blocks(soup):
    """
    Note: for some reason this works and few other attempted methods work
    Search for certain node types, find the first nodes siblings of the same type
    Add the first sibling and the other siblings to a list and return them
    """
    nodenames = body_block_nodenames()

    body_block_tags = []

    if not soup:
        return body_block_tags

    first_sibling_node = firstnn(soup.find_all())

    if first_sibling_node is None:
        return body_block_tags

    sibling_tags = first_sibling_node.find_next_siblings(nodenames)

    # Add the first component tag and the ResultSet tags together
    body_block_tags.append(first_sibling_node)

    for tag in sibling_tags:
        body_block_tags.append(tag)

    return body_block_tags