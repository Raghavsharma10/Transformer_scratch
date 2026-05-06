def shadow_reference(self, dispatcher, node):
        """
        Only simply make a reference to the value in the current scope,
        specifically for the FuncBase type.
        """

        # as opposed to the previous one, only add the value of the
        # identifier itself to the scope so that it becomes reserved.
        self.current_scope.reference(node.identifier.value)