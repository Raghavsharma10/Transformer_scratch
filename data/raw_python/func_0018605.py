def remove_insignificant_text_nodes(dom):
    """
    For html elements that should not have text nodes inside them, remove all
    whitespace. For elements that may have text, collapse multiple spaces to a
    single space.
    """
    nodes_to_remove = []
    for node in walk_dom(dom):
        if is_text(node):
            text = node.nodeValue
            if node.parentNode.tagName in _non_text_node_tags:
                nodes_to_remove.append(node)
            else:
                node.nodeValue = re.sub(r'\s+', ' ', text)
    for node in nodes_to_remove:
        remove_node(node)