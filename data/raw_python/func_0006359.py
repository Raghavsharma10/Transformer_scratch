def write_hits_in_events(hit_table_in, hit_table_out, events, start_hit_word=0, chunk_size=5000000, condition=None):
    '''Selects the hits that occurred in events and writes them to a pytable. This function reduces the in RAM operations and has to be
    used if the get_hits_in_events function raises a memory error. Also a condition can be set to select hits.

    Parameters
    ----------
    hit_table_in : pytable.table
    hit_table_out : pytable.table
        functions need to be able to write to hit_table_out
    events : array like
        defines the events to be written from hit_table_in to hit_table_out. They do not have to exists at all.
    start_hit_word: int
        Index of the first hit word to be analyzed. Used for speed up.
    chunk_size : int
        defines how many hits are analyzed in RAM. Bigger numbers increase the speed, too big numbers let the program crash with a memory error.
    condition : string
        A condition that is applied to the hits in numexpr style. Only if the expression evaluates to True the hit is taken.

    Returns
    -------
    start_hit_word: int
        Index of the last hit word analyzed. Used to speed up the next call of write_hits_in_events.
    '''
    if len(events) > 0:  # needed to avoid crash
        min_event = np.amin(events)
        max_event = np.amax(events)
        logging.debug("Write hits from hit number >= %d that exists in the selected %d events with %d <= event number <= %d into a new hit table." % (start_hit_word, len(events), min_event, max_event))
        table_size = hit_table_in.shape[0]
        iHit = 0
        for iHit in range(start_hit_word, table_size, chunk_size):
            hits = hit_table_in.read(iHit, iHit + chunk_size)
            last_event_number = hits[-1]['event_number']
            hit_table_out.append(get_hits_in_events(hits, events=events, condition=condition))
            if last_event_number > max_event:  # speed up, use the fact that the hits are sorted by event_number
                return iHit
    return start_hit_word