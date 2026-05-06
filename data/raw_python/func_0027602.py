def xpath_on_node(self, node, xpath, **kwargs):
        """
        Return result of performing the given XPath query on the given node.

        All known namespace prefix-to-URI mappings in the document are
        automatically included in the XPath invocation.

        If an empty/default namespace (i.e. None) is defined, this is
        converted to the prefix name '_' so it can be used despite empty
        namespace prefixes being unsupported by XPath.
        """
        if isinstance(node, etree._ElementTree):
            # Document node lxml.etree._ElementTree has no nsmap, lookup root
            root = self.get_impl_root(node)
            namespaces_dict = root.nsmap.copy()
        else:
            namespaces_dict = node.nsmap.copy()
        if 'namespaces' in kwargs:
            namespaces_dict.update(kwargs['namespaces'])
        # Empty namespace prefix is not supported, convert to '_' prefix
        if None in namespaces_dict:
            default_ns_uri = namespaces_dict.pop(None)
            namespaces_dict['_'] = default_ns_uri
        # Include XMLNS namespace if it's not already defined
        if not 'xmlns' in namespaces_dict:
            namespaces_dict['xmlns'] = nodes.Node.XMLNS_URI
        return node.xpath(xpath, namespaces=namespaces_dict)