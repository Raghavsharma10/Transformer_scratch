def find(self, name=None, ns_uri=None, first_only=False):
        """
        Find :class:`Element` node descendants of this node, with optional
        constraints to limit the results.

        :param name: limit results to elements with this name.
            If *None* or ``'*'`` all element names are matched.
        :type name: string or None
        :param ns_uri: limit results to elements within this namespace URI.
            If *None* all elements are matched, regardless of namespace.
        :type ns_uri: string or None
        :param bool first_only: if *True* only return the first result node
            or *None* if there is no matching node.

        :returns: a list of :class:`Element` nodes matching any given
            constraints, or a single node if ``first_only=True``.
        """
        if name is None:
            name = '*'  # Match all element names
        if ns_uri is None:
            ns_uri = '*'  # Match all namespaces
        impl_nodelist = self.adapter.find_node_elements(
            self.impl_node, name=name, ns_uri=ns_uri)
        if first_only:
            if impl_nodelist:
                return self.adapter.wrap_node(
                    impl_nodelist[0], self.adapter.impl_document, self.adapter)
            else:
                return None
        return self._convert_nodelist(impl_nodelist)