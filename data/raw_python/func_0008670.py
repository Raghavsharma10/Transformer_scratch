def to_python(self, value):
        """Converts the DB-stored value into a Python value."""
        if isinstance(value, base.StateWrapper):
            res = value
        else:
            if isinstance(value, base.State):
                state = value
            elif value is None:
                state = self.workflow.initial_state
            else:
                try:
                    state = self.workflow.states[value]
                except KeyError:
                    raise exceptions.ValidationError(self.error_messages['invalid'])
            res = base.StateWrapper(state, self.workflow)

        if res.state not in self.workflow.states:
            raise exceptions.ValidationError(self.error_messages['invalid'])

        return res