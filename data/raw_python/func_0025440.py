def __get_keys(self):
        """ Return the keys associated with this node by adding its key and then adding parent keys recursively. """
        keys = list()
        tree_node = self
        while tree_node is not None and tree_node.key is not None:
            keys.insert(0, tree_node.key)
            tree_node = tree_node.parent
        return keys