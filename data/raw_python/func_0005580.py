def flatten_subtree(self, name):  # noqa: D302
        r"""
        Flatten sub-tree.

        Nodes that have children and no data are merged with each child

        :param name: Ending hierarchy node whose sub-trees are to be
                     flattened
        :type  name: :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

         * RuntimeError (Node *[name]* not in tree)

        Using the same example tree created in
        :py:meth:`ptrie.Trie.add_nodes`::

            >>> from __future__ import print_function
            >>> import docs.support.ptrie_example
            >>> tobj = docs.support.ptrie_example.create_tree()
            >>> tobj.add_nodes([
            ...     {'name':'root.branch1.leaf1.subleaf2', 'data':[]},
            ...     {'name':'root.branch2.leaf1', 'data':'loren ipsum'},
            ...     {'name':'root.branch2.leaf1.another_subleaf1', 'data':[]},
            ...     {'name':'root.branch2.leaf1.another_subleaf2', 'data':[]}
            ... ])
            >>> print(str(tobj))
            root
            ├branch1 (*)
            │├leaf1
            ││├subleaf1 (*)
            ││└subleaf2
            │└leaf2 (*)
            │ └subleaf2
            └branch2
             └leaf1 (*)
              ├another_subleaf1
              └another_subleaf2
            >>> tobj.flatten_subtree('root.branch1.leaf1')
            >>> print(str(tobj))
            root
            ├branch1 (*)
            │├leaf1.subleaf1 (*)
            │├leaf1.subleaf2
            │└leaf2 (*)
            │ └subleaf2
            └branch2
             └leaf1 (*)
              ├another_subleaf1
              └another_subleaf2
            >>> tobj.flatten_subtree('root.branch2.leaf1')
            >>> print(str(tobj))
            root
            ├branch1 (*)
            │├leaf1.subleaf1 (*)
            │├leaf1.subleaf2
            │└leaf2 (*)
            │ └subleaf2
            └branch2
             └leaf1 (*)
              ├another_subleaf1
              └another_subleaf2
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        parent = self._db[name]["parent"]
        if (parent) and (not self._db[name]["data"]):
            children = self._db[name]["children"]
            for child in children:
                self._db[child]["parent"] = parent
            self._db[parent]["children"].remove(name)
            self._db[parent]["children"] = sorted(
                self._db[parent]["children"] + children
            )
            del self._db[name]