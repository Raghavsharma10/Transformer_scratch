def _add_transitions(mcs, field_name, workflow, attrs, implems=None):
        """Collect and enhance transition definitions to a workflow.

        Modifies the 'attrs' dict in-place.

        Args:
            field_name (str): name of the field transitions should update
            workflow (Workflow): workflow we're working on
            attrs (dict): dictionary of attributes to be updated.
            implems (ImplementationList): Implementation list from parent
                classes (optional)

        Returns:
            ImplementationList: The new implementation list for this field.
        """
        new_implems = ImplementationList(field_name, workflow)
        if implems:
            new_implems.load_parent_implems(implems)
        new_implems.transform(attrs)

        return new_implems