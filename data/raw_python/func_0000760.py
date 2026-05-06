def get_events_map():
    """ Prepare map of event subscribers.

    * Extends copies of BEFORE_EVENTS and AFTER_EVENTS maps with
        'set' action.
    * Returns map of {before/after: {action: event class(es)}}
    """
    from nefertari import events
    set_keys = ('create', 'update', 'replace', 'update_many', 'register')
    before_events = events.BEFORE_EVENTS.copy()
    before_events['set'] = [before_events[key] for key in set_keys]
    after_events = events.AFTER_EVENTS.copy()
    after_events['set'] = [after_events[key] for key in set_keys]
    return {
        'before': before_events,
        'after': after_events,
    }