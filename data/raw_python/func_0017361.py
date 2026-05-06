def _find_workflows(mcs, attrs):
        """Finds all occurrences of a workflow in the attributes definitions.

        Returns:
            dict(str => StateField): maps an attribute name to a StateField
                describing the related Workflow.
        """
        workflows = {}
        for attribute, value in attrs.items():
            if isinstance(value, Workflow):
                workflows[attribute] = StateField(value)
        return workflows