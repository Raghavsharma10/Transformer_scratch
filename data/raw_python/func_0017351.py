def add_implem(self, transition, attribute, function, **kwargs):
        """Add an implementation.

        Args:
            transition (Transition): the transition for which the implementation
                is added
            attribute (str): the name of the attribute where the implementation
                will be available
            function (callable): the actual implementation function
            **kwargs: extra arguments for the related ImplementationProperty.
        """
        implem = ImplementationProperty(
            field_name=self.state_field,
            transition=transition,
            workflow=self.workflow,
            implementation=function,
            **kwargs)
        self.implementations[transition.name] = implem
        self.transitions_at[transition.name] = attribute
        return implem