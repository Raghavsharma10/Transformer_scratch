def is_leaf(self, name):
        r"""
        Test if a node is a leaf node (node with no children).

        :param name: Node name
        :type  name: :ref:`NodeName`

        :rtype: boolean

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return not self._db[name]["children"]