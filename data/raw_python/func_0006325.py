def analyze_hits_per_scan_parameter(analyze_data, scan_parameters=None, chunk_size=50000):
    '''Takes the hit table and analyzes the hits per scan parameter

    Parameters
    ----------
    analyze_data : analysis.analyze_raw_data.AnalyzeRawData object with an opened hit file (AnalyzeRawData.out_file_h5) or a
    file name with the hit data given (AnalyzeRawData._analyzed_data_file)
    scan_parameters : list of strings:
        The names of the scan parameters to use
    chunk_size : int:
        The chunk size of one hit table read. The bigger the faster. Too big causes memory errors.
    Returns
    -------
    yields the analysis.analyze_raw_data.AnalyzeRawData for each scan parameter
    '''

    if analyze_data.out_file_h5 is None or analyze_data.out_file_h5.isopen == 0:
        in_hit_file_h5 = tb.open_file(analyze_data._analyzed_data_file, 'r+')
        close_file = True
    else:
        in_hit_file_h5 = analyze_data.out_file_h5
        close_file = False

    meta_data = in_hit_file_h5.root.meta_data[:]  # get the meta data table
    try:
        hit_table = in_hit_file_h5.root.Hits  # get the hit table
    except tb.NoSuchNodeError:
        logging.error('analyze_hits_per_scan_parameter needs a hit table, but no hit table found.')
        return

    meta_data_table_at_scan_parameter = analysis_utils.get_unique_scan_parameter_combinations(meta_data, scan_parameters=scan_parameters)
    parameter_values = analysis_utils.get_scan_parameters_table_from_meta_data(meta_data_table_at_scan_parameter, scan_parameters)
    event_number_ranges = analysis_utils.get_ranges_from_array(meta_data_table_at_scan_parameter['event_number'])  # get the event number ranges for the different scan parameter settings

    analysis_utils.index_event_number(hit_table)  # create a event_numer index to select the hits by their event number fast, no needed but important for speed up

    # variables for read speed up
    index = 0  # index where to start the read out of the hit table, 0 at the beginning, increased during looping
    best_chunk_size = chunk_size  # number of hits to copy to RAM during looping, the optimal chunk size is determined during looping

    # loop over the selected events
    for parameter_index, (start_event_number, stop_event_number) in enumerate(event_number_ranges):
        logging.info('Analyze hits for ' + str(scan_parameters) + ' = ' + str(parameter_values[parameter_index]))
        analyze_data.reset()  # resets the front end data of the last analysis step but not the options
        readout_hit_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
        # loop over the hits in the actual selected events with optimizations: determine best chunk size, start word index given
        for hits, index in analysis_utils.data_aligned_at_events(hit_table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=index, chunk_size=best_chunk_size):
            analyze_data.analyze_hits(hits, scan_parameter=False)  # analyze the selected hits in chunks
            readout_hit_len += hits.shape[0]
        best_chunk_size = int(1.5 * readout_hit_len) if int(1.05 * readout_hit_len) < chunk_size and int(1.05 * readout_hit_len) > 1e3 else chunk_size  # to increase the readout speed, estimated the number of hits for one read instruction
        file_name = " ".join(re.findall("[a-zA-Z0-9]+", str(scan_parameters))) + '_' + " ".join(re.findall("[a-zA-Z0-9]+", str(parameter_values[parameter_index])))
        analyze_data._create_additional_hit_data(safe_to_file=False)
        analyze_data._create_additional_cluster_data(safe_to_file=False)
        yield analyze_data, file_name

    if close_file:
        in_hit_file_h5.close()