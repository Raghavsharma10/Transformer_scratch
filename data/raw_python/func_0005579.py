def delete_subtree(self, nodes):  # noqa: D302
        r"""
        Delete nodes (and their sub-trees) from the tree.

        :param nodes: Node(s) to delete
        :type  nodes: :ref:`NodeName` or list of :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`nodes\` is not valid)

         * RuntimeError (Node *[node_name]* not in tree)

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
            >>> tobj.delete_subtree(['root.branch1.leaf1', 'root.branch2'])
            >>> print(tobj)
            root
            └branch1 (*)
             └leaf2 (*)
              └subleaf2
        """
        if self._validate_node_name(nodes):
            raise RuntimeError("Argument `nodes` is not valid")
        self._delete_subtree(nodes)