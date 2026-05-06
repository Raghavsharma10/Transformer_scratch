def _is_node_an_element(self, node):
        """
        Return True if the given node is an ElementTree Element, a fact that
        can be tricky to determine if the cElementTree implementation is
        used.
        """
        # Try the simplest approach first, works for plain old ElementTree
        if isinstance(node, BaseET.Element):
            return True
        # For cElementTree we need to be more cunning (or find a better way)
        if hasattr(node, 'makeelement') and isinstance(node.tag, basestring):
            return True