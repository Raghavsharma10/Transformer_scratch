def get_leafs(self, name):
        r"""
        Get the sub-tree leaf node(s).

        :param name: Sub-tree root node name
        :type  name: :ref:`NodeName`

        :rtype: list of :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return [node for node in self._get_subtree(name) if self.is_leaf(node)]