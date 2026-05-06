def nest(self, node, cls=None):
        """
        Create a new nested scope that is within this instance, binding
        the provided node to it.
        """

        if cls is None:
            cls = type(self)

        nested_scope = cls(node, self)
        self.children.append(nested_scope)
        return nested_scope