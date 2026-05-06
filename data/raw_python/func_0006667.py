def traverse_until_fixpoint(predicate, tree):
    """Traverses the tree again and again until it is not modified."""
    old_tree = None
    tree = simplify(tree)
    while tree and old_tree != tree:
        old_tree = tree
        tree = tree.traverse(predicate)
        if not tree:
            return None
        tree = simplify(tree)
    return tree