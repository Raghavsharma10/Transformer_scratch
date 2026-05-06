def make_root(self, name):  # noqa: D302
        r"""
        Make a sub-node the root node of the tree.

        All nodes not belonging to the sub-tree are deleted

        :param name: New root node name
        :type  name: :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not valid)

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
            >>> tobj.make_root('root.branch1')
            >>> print(tobj)
            root.branch1 (*)
            ├leaf1
            │└subleaf1 (*)
            └leaf2 (*)
             └subleaf2
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        if (name != self.root_name) and (self._node_in_tree(name)):
            for key in [node for node in self.nodes if node.find(name) != 0]:
                del self._db[key]
            self._db[name]["parent"] = ""
            self._root = name
            self._root_hierarchy_length = len(
                self.root_name.split(self._node_separator)
            )