def get_subtree(self, name):  # noqa: D302
        r"""
        Get all node names in a sub-tree.

        :param name: Sub-tree root node name
        :type  name: :ref:`NodeName`

        :rtype: list of :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)

        Using the same example tree created in
        :py:meth:`ptrie.Trie.add_nodes`::

            >>> from __future__ import print_function
            >>> import docs.support.ptrie_example, pprint
            >>> tobj = docs.support.ptrie_example.create_tree()
            >>> print(tobj)
            root
            ├branch1 (*)
            │├leaf1
            ││└subleaf1 (*)
            │└leaf2 (*)
            │ └subleaf2
            └branch2
            >>> pprint.pprint(tobj.get_subtree('root.branch1'))
            ['root.branch1',
             'root.branch1.leaf1',
             'root.branch1.leaf1.subleaf1',
             'root.branch1.leaf2',
             'root.branch1.leaf2.subleaf2']
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        return self._get_subtree(name)