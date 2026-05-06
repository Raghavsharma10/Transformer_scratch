def get_node(self, name):
        r"""
        Get a tree node structure.

        The structure is a dictionary with the following keys:

         * **parent** (*NodeName*) Parent node name, :code:`''` if the
           node is the root node

         * **children** (*list of NodeName*) Children node names, an
           empty list if node is a leaf

         * **data** (*list*) Node data, an empty list if node contains no data

        :param name: Node name
        :type  name: string

        :rtype: dictionary

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return self._db[name]