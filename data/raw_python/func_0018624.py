def distribute(node):
    """
    Wrap a copy of the given element around the contents of each of its
    children, removing the node in the process.
    """
    children = list(c for c in node.childNodes if is_element(c))
    unwrap(node)
    tag_name = node.tagName
    for c in children:
        wrap_inner(c, tag_name)