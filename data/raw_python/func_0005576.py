def collapse_subtree(self, name, recursive=True):  # noqa: D302
        r"""
        Collapse a sub-tree.

        Nodes that have a single child and no data are combined with their
        child as a single tree node

        :param name: Root of the sub-tree to collapse
        :type  name: :ref:`NodeName`

        :param recursive: Flag that indicates whether the collapse operation
                          is performed on the whole sub-tree (True) or whether
                          it stops upon reaching the first node where the
                          collapsing condition is not satisfied (False)
        :type  recursive: boolean

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Argument \`recursive\` is not valid)

         * RuntimeError (Node *[name]* not in tree)

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
            >>> tobj.collapse_subtree('root.branch1')
            >>> print(tobj)
            root
            ├branch1 (*)
            │├leaf1.subleaf1 (*)
            │└leaf2 (*)
            │ └subleaf2
            └branch2

        ``root.branch1.leaf1`` is collapsed because it only has one child
        (``root.branch1.leaf1.subleaf1``) and no data; ``root.branch1.leaf2``
        is not collapsed because although it has one child
        (``root.branch1.leaf2.subleaf2``) and this child does have data
        associated with it, :code:`'Hello world!'`
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        if not isinstance(recursive, bool):
            raise RuntimeError("Argument `recursive` is not valid")
        self._node_in_tree(name)
        self._collapse_subtree(name, recursive)