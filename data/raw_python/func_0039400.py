def acts_as_state_machine(cls):
    """
    a decorator which sets two properties on a class:
        * the 'current_state' property: a read-only property, returning the state machine's current state, as 'State' object
        * the 'states' property: a tuple of all valid state machine states, as 'State' objects
    class objects may use current_state and states freely
    :param cls:
    :return:
    """
    assert not hasattr(cls, 'current_state'), '{0} already has a "current_state" attribute!'.format(cls)
    assert not hasattr(cls, 'states'), '{0} already has a "states" attribute!'.format(cls)

    def get_states(obj):
        return StateInfo.get_states(obj.__class__)

    def is_transition_failure_handler(obj):
        return all([
            any([
                inspect.ismethod(obj),  # python2
                inspect.isfunction(obj),  # python3
            ]),
            getattr(obj, '___pystatemachine_is_transition_failure_handler', False),
        ])

    transition_failure_handlers = sorted(
        [value for name, value in inspect.getmembers(cls, is_transition_failure_handler)],
        key=lambda m: getattr(m, '___pystatemachine_transition_failure_handler_calling_sequence', 0),
    )
    setattr(cls, '___pystatemachine_transition_failure_handlers', transition_failure_handlers)
    cls.current_state = property(fget=StateInfo.get_current_state)
    cls.states = property(fget=get_states)
    return cls