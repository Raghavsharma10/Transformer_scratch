def _setup_states(state_definitions, prev=()):
    """Create a StateList object from a 'states' Workflow attribute."""
    states = list(prev)
    for state_def in state_definitions:
        if len(state_def) != 2:
            raise TypeError(
                "The 'state' attribute of a workflow should be "
                "a two-tuple of strings; got %r instead." % (state_def,)
            )
        name, title = state_def
        state = State(name, title)
        if any(st.name == name for st in states):
            # Replacing an existing state
            states = [state if st.name == name else st for st in states]
        else:
            states.append(state)
    return StateList(states)