def search_tree(self, name):  # noqa: D302
        r"""
        Search tree for all nodes with a specific name.

        :param name: Node name to search for
        :type  name: :ref:`NodeName`

        :raises: RuntimeError (Argument \`name\` is not valid)

        For example:

            >>> from __future__ import print_function
            >>> import pprint, ptrie
            >>> tobj = ptrie.Trie('/')
            >>> tobj.add_nodes([
            ...     {'name':'root', 'data':[]},
            ...     {'name':'root/anode', 'data':7},
            ...     {'name':'root/bnode', 'data':[]},
            ...     {'name':'root/cnode', 'data':[]},
            ...     {'name':'root/bnode/anode', 'data':['a', 'b', 'c']},
            ...     {'name':'root/cnode/anode/leaf', 'data':True}
            ... ])
            >>> print(tobj)
            root
            ├anode (*)
            ├bnode
            │└anode (*)
            └cnode
             └anode
              └leaf (*)
            >>> pprint.pprint(tobj.search_tree('anode'), width=40)
            ['root/anode',
             'root/bnode/anode',
             'root/cnode/anode',
             'root/cnode/anode/leaf']
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        return self._search_tree(name)