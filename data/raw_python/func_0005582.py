def get_data(self, name):
        r"""
        Get the data associated with a node.

        :param name: Node name
        :type  name: :ref:`NodeName`

        :rtype: any type or list of objects of any type

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return self._db[name]["data"]