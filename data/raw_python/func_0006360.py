def write_hits_in_event_range(hit_table_in, hit_table_out, event_start=None, event_stop=None, start_hit_word=0, chunk_size=5000000, condition=None):
    '''Selects the hits that occurred in given event range [event_start, event_stop[ and write them to a pytable. This function reduces the in RAM
       operations and has to be used if the get_data_in_event_range function raises a memory error. Also a condition can be set to select hits.

    Parameters
    ----------
    hit_table_in : pytable.table
    hit_table_out : pytable.table
        functions need to be able to write to hit_table_out
    event_start, event_stop : int, None
        start/stop event numbers. Stop event number is excluded. If None start/stop is set automatically.
    chunk_size : int
        defines how many hits are analyzed in RAM. Bigger numbers increase the speed, too big numbers let the program crash with a memory error.
    condition : string
        A condition that is applied to the hits in numexpr style. Only if the expression evaluates to True the hit is taken.
    Returns
    -------
    start_hit_word: int
        Index of the last hit word analyzed. Used to speed up the next call of write_hits_in_events.
    '''

    logging.debug('Write hits that exists in the given event range from + ' + str(event_start) + ' to ' + str(event_stop) + ' into a new hit table')
    table_size = hit_table_in.shape[0]
    for iHit in range(0, table_size, chunk_size):
        hits = hit_table_in.read(iHit, iHit + chunk_size)
        last_event_number = hits[-1]['event_number']
        selected_hits = get_data_in_event_range(hits, event_start=event_start, event_stop=event_stop)
        if condition is not None:
            # bad hack to be able to use numexpr
            for variable in set(re.findall(r'[a-zA-Z_]+', condition)):
                exec(variable + ' = hits[\'' + variable + '\']')
            selected_hits = selected_hits[ne.evaluate(condition)]
        hit_table_out.append(selected_hits)
        if last_event_number > event_stop:  # speed up, use the fact that the hits are sorted by event_number
            return iHit + chunk_size
    return start_hit_word