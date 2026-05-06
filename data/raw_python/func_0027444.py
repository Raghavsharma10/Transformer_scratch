def get_ns_info_from_node_name(self, name, impl_node):
        """
        Return a three-element tuple with the prefix, local name, and namespace
        URI for the given element/attribute name (in the context of the given
        node's hierarchy). If the name has no associated prefix or namespace
        information, None is return for those tuple members.
        """
        if '}' in name:
            ns_uri, name = name.split('}')
            ns_uri = ns_uri[1:]
            prefix = self.get_ns_prefix_for_uri(impl_node, ns_uri)
        elif ':' in name:
            prefix, name = name.split(':')
            ns_uri = self.get_ns_uri_for_prefix(impl_node, prefix)
            if ns_uri is None:
                raise exceptions.UnknownNamespaceException(
                    "Prefix '%s' does not have a defined namespace URI"
                    % prefix)
        else:
            prefix, ns_uri = None, None
        return prefix, name, ns_uri