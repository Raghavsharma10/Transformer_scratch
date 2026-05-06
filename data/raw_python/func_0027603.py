def _is_ns_in_ancestor(self, node, name, value):
        """
        Return True if the given namespace name/value is defined in an
        ancestor of the given node, meaning that the given node need not
        have its own attributes to apply that namespacing.
        """
        curr_node = self.get_node_parent(node)
        while curr_node.__class__ == etree._Element:
            if (hasattr(curr_node, 'nsmap')
                    and curr_node.nsmap.get(name) == value):
                return True
            for n, v in curr_node.attrib.items():
                if v == value and '{%s}' % nodes.Node.XMLNS_URI in n:
                    return True
            curr_node = self.get_node_parent(curr_node)
        return False