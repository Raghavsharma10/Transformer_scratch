def should_collect(self, value):
        """Decide whether a given value should be collected."""
        return (
            # decorated with @transition
            isinstance(value, TransitionWrapper)
            # Relates to a compatible transition
            and value.trname in self.workflow.transitions
            # Either not bound to a state field or bound to the current one
            and (not value.field or value.field == self.state_field))