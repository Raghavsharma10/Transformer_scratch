def get_node_children(self, name):
        r"""
        Get the list of children structures of a node.

        See :py:meth:`ptrie.Trie.get_node` for details about the structure

        :param name: Parent node name
        :type  name: :ref:`NodeName`

        :rtype: list

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return [self._db[child] for child in self._db[name]["children"]]