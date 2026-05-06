def remove_nesting(dom, tag_name):
    """
    Unwrap items in the node list that have ancestors with the same tag.
    """
    for node in dom.getElementsByTagName(tag_name):
        for ancestor in ancestors(node):
            if ancestor is node:
                continue
            if ancestor is dom.documentElement:
                break
            if ancestor.tagName == tag_name:
                unwrap(node)
                break