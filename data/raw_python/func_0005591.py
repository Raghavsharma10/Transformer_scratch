def print_node(self, name):  # noqa: D302
        r"""
        Print node information (parent, children and data).

        :param name: Node name
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
            >>> print(tobj.print_node('root.branch1'))
            Name: root.branch1
            Parent: root
            Children: leaf1, leaf2
            Data: [5, 7]
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        self._node_in_tree(name)
        node = self._db[name]
        children = (
            [self._split_node_name(child)[-1] for child in node["children"]]
            if node["children"]
            else node["children"]
        )
        data = (
            node["data"][0]
            if node["data"] and (len(node["data"]) == 1)
            else node["data"]
        )
        return (
            "Name: {node_name}\n"
            "Parent: {parent_name}\n"
            "Children: {children_list}\n"
            "Data: {node_data}".format(
                node_name=name,
                parent_name=node["parent"] if node["parent"] else None,
                children_list=", ".join(children) if children else None,
                node_data=data if data else None,
            )
        )