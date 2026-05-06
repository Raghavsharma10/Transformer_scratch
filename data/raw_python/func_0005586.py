def get_node_parent(self, name):
        r"""
        Get the parent structure of a node.

        See :py:meth:`ptrie.Trie.get_node` for details about the structure

        :param name: Child node name
        :type  name: :ref:`NodeName`

        :rtype: dictionary

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return self._db[self._db[name]["parent"]] if not self.is_root(name) else {}