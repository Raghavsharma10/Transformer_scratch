def _add_workflow(mcs, field_name, state_field, attrs):
        """Attach a workflow to the attribute list (create a StateProperty)."""
        attrs[field_name] = StateProperty(state_field.workflow, field_name)