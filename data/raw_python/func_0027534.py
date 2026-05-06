def set_ns_prefix(self, prefix, ns_uri):
        """
        Define a namespace prefix that will serve as shorthand for the given
        namespace URI in element names.

        :param string prefix: prefix that will serve as an alias for a
            the namespace URI.
        :param string ns_uri: namespace URI that will be denoted by the
            prefix.
        """
        self._add_ns_prefix_attr(self.impl_node, prefix, ns_uri)