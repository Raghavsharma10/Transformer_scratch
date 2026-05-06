def get_hits_in_events(hits_array, events, assume_sorted=True, condition=None):
    '''Selects the hits that occurred in events and optional selection criterion.
        If a event range can be defined use the get_data_in_event_range function. It is much faster.

    Parameters
    ----------
    hits_array : numpy.array
    events : array
    assume_sorted : bool
        Is true if the events to select are sorted from low to high value. Increases speed by 35%.
    condition : string
        A condition that is applied to the hits in numexpr. Only if the expression evaluates to True the hit is taken.

    Returns
    -------
    numpy.array
        hit array with the hits in events.
    '''

    logging.debug("Calculate hits that exists in the given %d events." % len(events))
    if assume_sorted:
        events, _ = reduce_sorted_to_intersect(events, hits_array['event_number'])  # reduce the event number range to the max min event number of the given hits to save time
        if events.shape[0] == 0:  # if there is not a single selected hit
            return hits_array[0:0]
    try:
        if assume_sorted:
            selection = analysis_utils.in1d_events(hits_array['event_number'], events)
        else:
            logging.warning('Events are usually sorted. Are you sure you want this?')
            selection = np.in1d(hits_array['event_number'], events)
        if condition is None:
            hits_in_events = hits_array[selection]
        else:
            # bad hack to be able to use numexpr
            for variable in set(re.findall(r'[a-zA-Z_]+', condition)):
                exec(variable + ' = hits_array[\'' + variable + '\']')

            hits_in_events = hits_array[ne.evaluate(condition + ' & selection')]
    except MemoryError:
        logging.error('There are too many hits to do in RAM operations. Consider decreasing chunk size and use the write_hits_in_events function instead.')
        raise MemoryError
    return hits_in_events