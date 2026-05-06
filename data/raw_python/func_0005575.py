def add_nodes(self, nodes):  # noqa: D302
        r"""
        Add nodes to tree.

        :param nodes: Node(s) to add with associated data. If there are
                      several list items in the argument with the same node
                      name the resulting node data is a list with items
                      corresponding to the data of each entry in the argument
                      with the same node name, in their order of appearance,
                      in addition to any existing node data if the node is
                      already present in the tree
        :type  nodes: :ref:`NodesWithData`

        :raises:
         * RuntimeError (Argument \`nodes\` is not valid)

         * ValueError (Illegal node name: *[node_name]*)

        For example:

        .. =[=cog
        .. import docs.support.incfile
        .. docs.support.incfile.incfile('ptrie_example.py', cog.out)
        .. =]=
        .. code-block:: python

            # ptrie_example.py
            import ptrie

            def create_tree():
                tobj = ptrie.Trie()
                tobj.add_nodes([
                    {'name':'root.branch1', 'data':5},
                    {'name':'root.branch1', 'data':7},
                    {'name':'root.branch2', 'data':[]},
                    {'name':'root.branch1.leaf1', 'data':[]},
                    {'name':'root.branch1.leaf1.subleaf1', 'data':333},
                    {'name':'root.branch1.leaf2', 'data':'Hello world!'},
                    {'name':'root.branch1.leaf2.subleaf2', 'data':[]},
                ])
                return tobj

        .. =[=end=]=

        .. code-block:: python

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

            >>> tobj.get_data('root.branch1')
            [5, 7]
        """
        self._validate_nodes_with_data(nodes)
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # Create root node (if needed)
        if not self.root_name:
            self._set_root_name(nodes[0]["name"].split(self._node_separator)[0].strip())
            self._root_hierarchy_length = len(
                self.root_name.split(self._node_separator)
            )
            self._create_node(name=self.root_name, parent="", children=[], data=[])
        # Process new data
        for node_dict in nodes:
            name, data = node_dict["name"], node_dict["data"]
            if name not in self._db:
                # Validate node name (root of new node same as tree root)
                if not name.startswith(self.root_name + self._node_separator):
                    raise ValueError("Illegal node name: {0}".format(name))
                self._create_intermediate_nodes(name)
            self._db[name]["data"] += copy.deepcopy(
                data
                if isinstance(data, list) and data
                else ([] if isinstance(data, list) else [data])
            )