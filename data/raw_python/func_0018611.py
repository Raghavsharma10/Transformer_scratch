def insert_or_append(parent, node, next_sibling):
    """
    Insert the node before next_sibling. If next_sibling is None, append the node last instead.
    """
    # simple insert
    if next_sibling:
        parent.insertBefore(node, next_sibling)
    else:
        parent.appendChild(node)