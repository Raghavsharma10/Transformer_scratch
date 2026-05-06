def in_tree(self, name):
        r"""
        Test if a node is in the tree.

        :param name: Node name to search for
        :type  name: :ref:`NodeName`

        :rtype: boolean

        :raises: RuntimeError (Argument \`name\` is not valid)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        return name in self._db