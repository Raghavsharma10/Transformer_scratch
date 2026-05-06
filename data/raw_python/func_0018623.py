def merge_adjacent(dom, tag_name):
    """
    Merge all adjacent tags with the specified tag name.
    Return the number of merges performed.
    """
    for node in dom.getElementsByTagName(tag_name):
        prev_sib = node.previousSibling
        if prev_sib and prev_sib.nodeName == node.tagName:
            for child in list(node.childNodes):
                prev_sib.appendChild(child)
            remove_node(node)