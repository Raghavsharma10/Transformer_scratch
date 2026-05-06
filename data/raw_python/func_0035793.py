def istr_type(istr):
    """
    Given an "ion" specification, determine its "type", e.g. 1D, Events, etc.
    """
    data = set(i.rstrip('0123456789') for i in tokens(istr))
    has_events = not data.isdisjoint(istr_type_evts)
    has_2d = not data.isdisjoint(istr_type_2d)
    has_1d = data.difference(istr_type_evts).difference(istr_type_2d) != set()

    if has_events and not (has_1d or has_2d):
        return 'events'
    elif has_1d and not has_events:
        return '1d'
    elif has_2d and not (has_events or has_1d):
        return '2d'
    else:
        return None