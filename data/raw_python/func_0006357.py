def get_hits_of_scan_parameter(input_file_hits, scan_parameters=None, try_speedup=False, chunk_size=10000000):
    '''Takes the hit table of a hdf5 file and returns hits in chunks for each unique combination of scan_parameters.
    Yields the hits in chunks, since they usually do not fit into memory.

    Parameters
    ----------
    input_file_hits : pytable hdf5 file
        Has to include a hits node
    scan_parameters : iterable with strings
    try_speedup : bool
        If true a speed up by searching for the event numbers in the data is done. If the event numbers are not in the data
        this slows down the search.
    chunk_size : int
        How many rows of data are read into ram.

    Returns
    -------
    Yields tuple, numpy.array
        Actual scan parameter tuple, hit array with the hits of a chunk of the given scan parameter tuple
    '''

    with tb.open_file(input_file_hits, mode="r+") as in_file_h5:
        hit_table = in_file_h5.root.Hits
        meta_data = in_file_h5.root.meta_data[:]
        meta_data_table_at_scan_parameter = get_unique_scan_parameter_combinations(meta_data, scan_parameters=scan_parameters)
        parameter_values = get_scan_parameters_table_from_meta_data(meta_data_table_at_scan_parameter, scan_parameters)
        event_number_ranges = get_ranges_from_array(meta_data_table_at_scan_parameter['event_number'])  # get the event number ranges for the different scan parameter settings
        index_event_number(hit_table)  # create a event_numer index to select the hits by their event number fast, no needed but important for speed up
#
        # variables for read speed up
        index = 0  # index where to start the read out of the hit table, 0 at the beginning, increased during looping
        best_chunk_size = chunk_size  # number of hits to copy to RAM during looping, the optimal chunk size is determined during looping

        # loop over the selected events
        for parameter_index, (start_event_number, stop_event_number) in enumerate(event_number_ranges):
            logging.debug('Read hits for ' + str(scan_parameters) + ' = ' + str(parameter_values[parameter_index]))

            readout_hit_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
            # loop over the hits in the actual selected events with optimizations: determine best chunk size, start word index given
            for hits, index in data_aligned_at_events(hit_table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=index, try_speedup=try_speedup, chunk_size=best_chunk_size):
                yield parameter_values[parameter_index], hits
                readout_hit_len += hits.shape[0]
            best_chunk_size = int(1.5 * readout_hit_len) if int(1.05 * readout_hit_len) < chunk_size and int(1.05 * readout_hit_len) > 1e3 else chunk_size