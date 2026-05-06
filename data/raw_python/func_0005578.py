def delete_prefix(self, name):  # noqa: D302
        r"""
        Delete hierarchy levels from all nodes in the tree.

        :param nodes: Prefix to delete
        :type  nodes: :ref:`NodeName`

        :raises:
         * RuntimeError (Argument \`name\` is not a valid prefix)

         * RuntimeError (Argument \`name\` is not valid)

        For example:

            >>> from __future__ import print_function
            >>> import ptrie
            >>> tobj = ptrie.Trie('/')
            >>> tobj.add_nodes([
            ...     {'name':'hello/world/root', 'data':[]},
            ...     {'name':'hello/world/root/anode', 'data':7},
            ...     {'name':'hello/world/root/bnode', 'data':8},
            ...     {'name':'hello/world/root/cnode', 'data':False},
            ...     {'name':'hello/world/root/bnode/anode', 'data':['a', 'b']},
            ...     {'name':'hello/world/root/cnode/anode/leaf', 'data':True}
            ... ])
            >>> tobj.collapse_subtree('hello', recursive=False)
            >>> print(tobj)
            hello/world/root
            ├anode (*)
            ├bnode (*)
            │└anode (*)
            └cnode (*)
             └anode
              └leaf (*)
            >>> tobj.delete_prefix('hello/world')
            >>> print(tobj)
            root
            ├anode (*)
            ├bnode (*)
            │└anode (*)
            └cnode (*)
             └anode
              └leaf (*)
        """
        if self._validate_node_name(name):
            raise RuntimeError("Argument `name` is not valid")
        if (not self.root_name.startswith(name)) or (self.root_name == name):
            raise RuntimeError("Argument `name` is not a valid prefix")
        self._delete_prefix(name)