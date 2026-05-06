def filter(self, local_name=None, name=None, ns_uri=None, node_type=None,
            filter_fn=None, first_only=False):
        """
        Apply filters to the set of nodes in this list.

        :param local_name: a local name used to filter the nodes.
        :type local_name: string or None
        :param name: a name used to filter the nodes.
        :type name: string or None
        :param ns_uri: a namespace URI used to filter the nodes.
            If *None* all nodes are returned regardless of namespace.
        :type ns_uri: string or None
        :param node_type: a node type definition used to filter the nodes.
        :type node_type: int node type constant, class, or None
        :param filter_fn: an arbitrary function to filter nodes in this list.
            This function must accept a single :class:`Node` argument and
            return a bool indicating whether to include the node in the
            filtered results.

            .. note:: if ``filter_fn`` is provided all other filter arguments
                are ignore.
        :type filter_fn: function or None

        :return: the type of the return value depends on the value of the
            ``first_only`` parameter and how many nodes match the filter:

            - if ``first_only=False`` return a :class:`NodeList` of filtered
              nodes, which will be empty if there are no matching nodes.
            - if ``first_only=True`` and at least one node matches,
              return the first matching :class:`Node`
            - if ``first_only=True`` and there are no matching nodes,
              return *None*
        """
        # Build our own filter function unless a custom function is provided
        if filter_fn is None:
            def filter_fn(n):
                # Test node type first in case other tests require this type
                if node_type is not None:
                    # Node type can be specified as an integer constant (e.g.
                    # ELEMENT_NODE) or a class.
                    if isinstance(node_type, int):
                        if not n.is_type(node_type):
                            return False
                    elif n.__class__ != node_type:
                        return False
                if name is not None and n.name != name:
                    return False
                if local_name is not None and n.local_name != local_name:
                    return False
                if ns_uri is not None and n.ns_uri != ns_uri:
                    return False
                return True
        # Filter nodes
        nodelist = filter(filter_fn, self)
        # If requested, return just the first node (or None if no nodes)
        if first_only:
            return nodelist[0] if nodelist else None
        else:
            return NodeList(nodelist)