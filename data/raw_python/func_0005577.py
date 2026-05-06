def copy_subtree(self, source_node, dest_node):  # noqa: D302
        r"""
        Copy a sub-tree from one sub-node to another.

        Data is added if some nodes of the source sub-tree exist in the
        destination sub-tree

        :param source_name: Root node of the sub-tree to copy from
        :type  source_name: :ref:`NodeName`

        :param dest_name: Root node of the sub-tree to copy to
        :type  dest_name: :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`dest_node\` is not valid)

         * RuntimeError (Argument \`source_node\` is not valid)

         * RuntimeError (Illegal root in destination node)

         * RuntimeError (Node *[source_node]* not in tree)

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
            >>> tobj.copy_subtree('root.branch1', 'root.branch3')
            >>> print(tobj)
            root
            ├branch1 (*)
            │├leaf1
            ││└subleaf1 (*)
            │└leaf2 (*)
            │ └subleaf2
            ├branch2
            └branch3 (*)
             ├leaf1
             │└subleaf1 (*)
             └leaf2 (*)
              └subleaf2
        """
        if self._validate_node_name(source_node):
            raise RuntimeError("Argument `source_node` is not valid")
        if self._validate_node_name(dest_node):
            raise RuntimeError("Argument `dest_node` is not valid")
        if source_node not in self._db:
            raise RuntimeError("Node {0} not in tree".format(source_node))
        if not dest_node.startswith(self.root_name + self._node_separator):
            raise RuntimeError("Illegal root in destination node")
        for node in self._get_subtree(source_node):
            self._db[node.replace(source_node, dest_node, 1)] = {
                "parent": self._db[node]["parent"].replace(source_node, dest_node, 1),
                "children": [
                    child.replace(source_node, dest_node, 1)
                    for child in self._db[node]["children"]
                ],
                "data": copy.deepcopy(self._db[node]["data"]),
            }
        self._create_intermediate_nodes(dest_node)
        parent = self._node_separator.join(dest_node.split(self._node_separator)[:-1])
        self._db[dest_node]["parent"] = parent
        self._db[parent]["children"] = sorted(
            self._db[parent]["children"] + [dest_node]
        )