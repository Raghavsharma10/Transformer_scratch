def event(from_states=None, to_state=None):
    """ a decorator for transitioning from certain states to a target state. must be used on bound methods of a class
    instance, only. """
    from_states_tuple = (from_states, ) if isinstance(from_states, State) else tuple(from_states or [])
    if not len(from_states_tuple) >= 1:
        raise ValueError()
    if not all(isinstance(state, State) for state in from_states_tuple):
        raise TypeError()
    if not isinstance(to_state, State):
        raise TypeError()

    def wrapper(wrapped):

        @functools.wraps(wrapped)
        def transition(instance, *a, **kw):
            if instance.current_state not in from_states_tuple:
                raise InvalidStateTransition()
            try:
                result = wrapped(instance, *a, **kw)
            except Exception as error:
                error_handlers = getattr(instance, '___pystatemachine_transition_failure_handlers', [])
                for error_handler in error_handlers:
                    error_handler(instance, wrapped, instance.current_state, to_state, error)
                if not error_handlers:
                    raise error
            else:
                StateInfo.set_current_state(instance, to_state)
                return result

        return transition

    return wrapper