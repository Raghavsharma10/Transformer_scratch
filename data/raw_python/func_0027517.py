def child(self, local_name=None, name=None, ns_uri=None, node_type=None,
            filter_fn=None):
        """
        :return: the first child node matching the given constraints, or \
                 *None* if there are no matching child nodes.

        Delegates to :meth:`NodeList.filter`.
        """
        return self.children(name=name, local_name=local_name, ns_uri=ns_uri,
            node_type=node_type, filter_fn=filter_fn, first_only=True)