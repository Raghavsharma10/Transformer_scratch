def _find_workflows(mcs, attrs):
        """Find workflow definition(s) in a WorkflowEnabled definition.

        This method overrides the default behavior from xworkflows in order to
        use our custom StateField objects.
        """
        workflows = {}
        for k, v in attrs.items():
            if isinstance(v, StateField):
                workflows[k] = v
        return workflows