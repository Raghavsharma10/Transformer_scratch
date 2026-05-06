def deconstruct(self):
        """Deconstruction for migrations.

        Return a simpler object (_SerializedWorkflow), since our Workflows
        are rather hard to serialize: Django doesn't like deconstructing
        metaclass-built classes.
        """
        name, path, args, kwargs = super(StateField, self).deconstruct()

        # We want to display the proper class name, which isn't available
        # at the same point for _SerializedWorkflow and Workflow.
        if isinstance(self.workflow, _SerializedWorkflow):
            workflow_class_name = self.workflow._name
        else:
            workflow_class_name = self.workflow.__class__.__name__

        kwargs['workflow'] = _SerializedWorkflow(
            name=workflow_class_name,
            initial_state=str(self.workflow.initial_state.name),
            states=[str(st.name) for st in self.workflow.states],
        )
        del kwargs['choices']
        del kwargs['default']
        return name, path, args, kwargs