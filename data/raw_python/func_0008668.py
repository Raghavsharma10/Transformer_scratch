def render(self, name, value, attrs=None, *args, **kwargs):
        """Handle a few expected values for rendering the current choice.

        Extracts the state name from StateWrapper and State object.
        """
        if isinstance(value, base.StateWrapper):
            state_name = value.state.name
        elif isinstance(value, base.State):
            state_name = value.name
        else:
            state_name = str(value)
        return super(StateSelect, self).render(name, state_name, attrs, *args, **kwargs)