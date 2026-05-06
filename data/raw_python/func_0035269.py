def register_reference(self, dispatcher, node):
        """
        Register this identifier to the current scope, and mark it as
        referenced in the current scope.
        """

        # the identifier node itself will be mapped to the current scope
        # for the resolve to work
        # This should probably WARN about the node object being already
        # assigned to an existing scope that isn't current_scope.
        self.identifiers[node] = self.current_scope
        self.current_scope.reference(node.value)