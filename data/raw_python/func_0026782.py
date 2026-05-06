def get_node_parent_class(node):
    """Supposes that node is a mongoengine field in a class and tries to
    get its parent class"""

    while node.parent:  # pragma no branch
        if isinstance(node, ClassDef):
            return node

        node = node.parent