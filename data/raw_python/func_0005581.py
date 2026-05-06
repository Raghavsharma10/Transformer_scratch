def get_children(self, name):
        r"""
        Get the children node names of a node.

        :param name: Parent node name
        :type  name: :ref:`NodeName`

        :rtype: list of :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return sorted(self._db[name]["children"])