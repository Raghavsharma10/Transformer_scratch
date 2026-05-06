def data_aligned_at_events(table, start_event_number=None, stop_event_number=None, start_index=None, stop_index=None, chunk_size=10000000, try_speedup=False, first_event_aligned=True, fail_on_missing_events=True):
    '''Takes the table with a event_number column and returns chunks with the size up to chunk_size. The chunks are chosen in a way that the events are not splitted.
    Additional parameters can be set to increase the readout speed. Events between a certain range can be selected.
    Also the start and the stop indices limiting the table size can be specified to improve performance.
    The event_number column must be sorted.
    In case of try_speedup is True, it is important to create an index of event_number column with pytables before using this function. Otherwise the queries are slowed down.

    Parameters
    ----------
    table : pytables.table
        The data.
    start_event_number : int
        The retruned data contains events with event number >= start_event_number. If None, no limit is set.
    stop_event_number : int
        The retruned data contains events with event number < stop_event_number. If None, no limit is set.
    start_index : int
        Start index of data. If None, no limit is set.
    stop_index : int
        Stop index of data. If None, no limit is set.
    chunk_size : int
        Maximum chunk size per read.
    try_speedup : bool
        If True, try to reduce the index range to read by searching for the indices of start and stop event number. If these event numbers are usually
        not in the data this speedup can even slow down the function!

    The following parameters are not used when try_speedup is True:

    first_event_aligned : bool
        If True, assuming that the first event is aligned to the data chunk and will be added. If False, the lowest event number of the first chunk will not be read out.
    fail_on_missing_events : bool
        If True, an error is given when start_event_number or stop_event_number is not part of the data.

    Returns
    -------
    Iterator of tuples
        Data of the actual data chunk and start index for the next chunk.

    Example
    -------
    start_index = 0
    for scan_parameter in scan_parameter_range:
        start_event_number, stop_event_number = event_select_function(scan_parameter)
        for data, start_index in data_aligned_at_events(table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=start_index):
            do_something(data)

    for data, index in data_aligned_at_events(table):
        do_something(data)
    '''
    # initialize variables
    start_index_known = False
    stop_index_known = False
    start_index = 0 if start_index is None else start_index
    stop_index = table.nrows if stop_index is None else stop_index
    if stop_index < start_index:
        raise InvalidInputError('Invalid start/stop index')
    table_max_rows = table.nrows
    if stop_event_number is not None and start_event_number is not None and stop_event_number < start_event_number:
        raise InvalidInputError('Invalid start/stop event number')

    # set start stop indices from the event numbers for fast read if possible; not possible if the given event number does not exist in the data stream
    if try_speedup and table.colindexed["event_number"]:
        if start_event_number is not None:
            start_condition = 'event_number==' + str(start_event_number)
            start_indices = table.get_where_list(start_condition, start=start_index, stop=stop_index)
            if start_indices.shape[0] != 0:  # set start index if possible
                start_index = start_indices[0]
                start_index_known = True

        if stop_event_number is not None:
            stop_condition = 'event_number==' + str(stop_event_number)
            stop_indices = table.get_where_list(stop_condition, start=start_index, stop=stop_index)
            if stop_indices.shape[0] != 0:  # set the stop index if possible, stop index is excluded
                stop_index = stop_indices[0]
                stop_index_known = True

    if start_index_known and stop_index_known and start_index + chunk_size >= stop_index:  # special case, one read is enough, data not bigger than one chunk and the indices are known
        yield table.read(start=start_index, stop=stop_index), stop_index
    else:  # read data in chunks, chunks do not divide events, abort if stop_event_number is reached

        # search for begin
        current_start_index = start_index
        if start_event_number is not None:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                last_event_in_chunk = array_chunk["event_number"][-1]

                if last_event_in_chunk < start_event_number:
                    current_start_index = current_start_index + chunk_size  # not there yet, continue to next read (assuming sorted events)
                else:
                    first_event_in_chunk = array_chunk["event_number"][0]
#                     if stop_event_number is not None and first_event_in_chunk >= stop_event_number and start_index != 0 and start_index == current_start_index:
#                         raise InvalidInputError('The stop event %d is missing. Change stop_event_number.' % stop_event_number)
                    if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                        raise InvalidInputError('Chunk size too small. Increase chunk size to fit full event.')

                    if not first_event_aligned and first_event_in_chunk == start_event_number and start_index != 0 and start_index == current_start_index:  # first event in first chunk not aligned at index 0, so take next event
                        if fail_on_missing_events:
                            raise InvalidInputError('The start event %d is missing. Change start_event_number.' % start_event_number)
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number + 1, side='left')
                    elif fail_on_missing_events and first_event_in_chunk > start_event_number and start_index == current_start_index:
                        raise InvalidInputError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    elif first_event_aligned and first_event_in_chunk == start_event_number and start_index == current_start_index:
                        chunk_start_index = 0
                    else:
                        chunk_start_index = np.searchsorted(array_chunk["event_number"], start_event_number, side='left')
                        if fail_on_missing_events and array_chunk["event_number"][chunk_start_index] != start_event_number and start_index == current_start_index:
                            raise InvalidInputError('The start event %d is missing. Change start_event_number.' % start_event_number)
#                     if fail_on_missing_events and ((start_index == current_start_index and chunk_start_index == 0 and start_index != 0 and not first_event_aligned) or array_chunk["event_number"][chunk_start_index] != start_event_number):
#                         raise InvalidInputError('The start event %d is missing. Change start_event_number.' % start_event_number)
                    current_start_index = current_start_index + chunk_start_index  # calculate index for next loop
                    break
        elif not first_event_aligned and start_index != 0:
            while current_start_index < stop_index:
                current_stop_index = min(current_start_index + chunk_size, stop_index)
                array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
                first_event_in_chunk = array_chunk["event_number"][0]
                last_event_in_chunk = array_chunk["event_number"][-1]

                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise InvalidInputError('Chunk size too small. Increase chunk size to fit full event.')

                chunk_start_index = np.searchsorted(array_chunk["event_number"], first_event_in_chunk + 1, side='left')
                current_start_index = current_start_index + chunk_start_index
                if not first_event_in_chunk == last_event_in_chunk:
                    break

        # data loop
        while current_start_index < stop_index:
            current_stop_index = min(current_start_index + chunk_size, stop_index)
            array_chunk = table.read(start=current_start_index, stop=current_stop_index)  # stop index is exclusive, so add 1
            first_event_in_chunk = array_chunk["event_number"][0]
            last_event_in_chunk = array_chunk["event_number"][-1]

            chunk_start_index = 0

            if stop_event_number is None:
                if current_stop_index == table_max_rows:
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')
            else:
                if last_event_in_chunk >= stop_event_number:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], stop_event_number, side='left')
                elif current_stop_index == table_max_rows:  # this will also add the last event of the table
                    chunk_stop_index = array_chunk.shape[0]
                else:
                    chunk_stop_index = np.searchsorted(array_chunk["event_number"], last_event_in_chunk, side='left')

            nrows = chunk_stop_index - chunk_start_index
            if nrows == 0:
                if array_chunk.shape[0] == chunk_size and first_event_in_chunk == last_event_in_chunk:
                    raise InvalidInputError('Chunk size too small to fit event. Data corruption possible. Increase chunk size to read full event.')
                elif chunk_start_index == 0:  # not increasing current_start_index
                    return
                elif stop_event_number is not None and last_event_in_chunk >= stop_event_number:
                    return
            else:
                yield array_chunk[chunk_start_index:chunk_stop_index], current_start_index + nrows + chunk_start_index

            current_start_index = current_start_index + nrows + chunk_start_index