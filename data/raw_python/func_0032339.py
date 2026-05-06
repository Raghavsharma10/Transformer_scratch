def get_attrib(self, et_node, prefixed_attrib):
        """Get a prefixed attribute like 'rdf:resource' from ET node."""
        prefix, attrib = prefixed_attrib.split(':')
        return et_node.get('{{{0}}}{1}'.format(self.namespaces[prefix],
                                               attrib))