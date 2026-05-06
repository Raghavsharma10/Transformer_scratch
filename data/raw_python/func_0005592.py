def rename_node(self, name, new_name):  # noqa: D302
        r"""
        Rename a tree node.

        It is typical to have a root node name with more than one hierarchy
        level after using :py:meth:`ptrie.Trie.make_root`. In this instance the
        root node *can* be renamed as long as the new root name has the same or
        less hierarchy levels as the existing root name

        :param name: Node name to rename
        :type  name: :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Argument \`new_name\` has an illegal root node)

         * RuntimeError (Argument \`new_name\` is an illegal root node name)

         * RuntimeError (Argument \`new_name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)

         * RuntimeError (Node *[new_name]* already exists)

        Using the same example tree created in
        :py:meth:`ptrie.Trie.add_nodes`::

            >>> from __future__ import print_function
            >>> import docs.support.ptrie_example
            >>> tobj = docs.support.ptrie_example.create_tree()
            >>> print(tobj)
            root
            ├branch1 (*)
            │├leaf1
            ││└subleaf1 (*)
            │└leaf2 (*)
            │ └subleaf2
            └branch2
            >>> tobj.rename_node(
            ...     'root.branch1.leaf1',
            ...     'root.branch1.mapleleaf1'
            ... )
            >>> print(tobj)
            root
            ├branch1 (*)
            │├leaf2 (*)
            ││└subleaf2
            │└mapleleaf1
            │ └subleaf1 (*)
            └branch2
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        if self._validate_node_name(new_name):
            raise RuntimeError("Argument `new_name` is not valid")
        self._node_in_tree(name)
        if self.in_tree(new_name) and (name != self.root_name):
            raise RuntimeError("Node {0} already exists".format(new_name))
        sep = self._node_separator
        if (name.split(sep)[:-1] != new_name.split(sep)[:-1]) and (
            name != self.root_name
        ):
            raise RuntimeError("Argument `new_name` has an illegal root node")
        old_hierarchy_length = len(name.split(self._node_separator))
        new_hierarchy_length = len(new_name.split(self._node_separator))
        if (name == self.root_name) and (old_hierarchy_length < new_hierarchy_length):
            raise RuntimeError("Argument `new_name` is an illegal root node name")
        self._rename_node(name, new_name)