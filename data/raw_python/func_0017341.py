def _setup_transitions(tdef, states, prev=()):
    """Create a TransitionList object from a 'transitions' Workflow attribute.

    Args:
        tdef: list of transition definitions
        states (StateList): already parsed state definitions.
        prev (TransitionList): transition definitions from a parent.

    Returns:
        TransitionList: the list of transitions defined in the 'tdef' argument.
    """
    trs = list(prev)
    for transition in tdef:
        if len(transition) == 3:
            (name, source, target) = transition
            if is_string(source) or isinstance(source, State):
                source = [source]
            source = [states[src] for src in source]
            target = states[target]
            tr = Transition(name, source, target)
        else:
            raise TypeError(
                "Elements of the 'transition' attribute of a "
                "workflow should be three-tuples; got %r instead." % (transition,)
            )

        if any(prev_tr.name == tr.name for prev_tr in trs):
            # Replacing an existing state
            trs = [tr if prev_tr.name == tr.name else prev_tr for prev_tr in trs]
        else:
            trs.append(tr)
    return TransitionList(trs)