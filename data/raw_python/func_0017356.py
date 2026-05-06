def _may_override(self, implem, other):
        """Checks whether an ImplementationProperty may override an attribute."""
        if isinstance(other, ImplementationProperty):
            # Overriding another custom implementation for the same transition
            # and field
            return (other.transition == implem.transition and other.field_name == self.state_field)

        elif isinstance(other, TransitionWrapper):
            # Overriding the definition that led to adding the current
            # ImplementationProperty.
            return (
                other.trname == implem.transition.name
                and (not other.field or other.field == self.state_field)
                and other.func == implem.implementation)

        return False