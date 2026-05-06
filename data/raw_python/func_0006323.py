def analyze_cluster_size_per_scan_parameter(input_file_hits, output_file_cluster_size, parameter='GDAC', max_chunk_size=10000000, overwrite_output_files=False, output_pdf=None):
    ''' This method takes multiple hit files and determines the cluster size for different scan parameter values of

     Parameters
    ----------
    input_files_hits: string
    output_file_cluster_size: string
        The data file with the results
    parameter: string
        The name of the parameter to separate the data into (e.g.: PlsrDAC)
    max_chunk_size: int
        the maximum chunk size used during read, if too big memory error occurs, if too small analysis takes longer
    overwrite_output_files: bool
        Set to true to overwrite the output file if it already exists
    output_pdf: PdfPages
        PdfPages file object, if none the plot is printed to screen, if False nothing is printed
    '''
    logging.info('Analyze the cluster sizes for different ' + parameter + ' settings for ' + input_file_hits)
    if os.path.isfile(output_file_cluster_size) and not overwrite_output_files:  # skip analysis if already done
        logging.info('Analyzed cluster size file ' + output_file_cluster_size + ' already exists. Skip cluster size analysis.')
    else:
        with tb.open_file(output_file_cluster_size, mode="w") as out_file_h5:  # file to write the data into
            filter_table = tb.Filters(complib='blosc', complevel=5, fletcher32=False)  # compression of the written data
            parameter_goup = out_file_h5.create_group(out_file_h5.root, parameter, title=parameter)  # note to store the data
            cluster_size_total = None  # final array for the cluster size per GDAC
            with tb.open_file(input_file_hits, mode="r+") as in_hit_file_h5:  # open the actual hit file
                meta_data_array = in_hit_file_h5.root.meta_data[:]
                scan_parameter = analysis_utils.get_scan_parameter(meta_data_array)  # get the scan parameters
                if scan_parameter:  # if a GDAC scan parameter was used analyze the cluster size per GDAC setting
                    scan_parameter_values = scan_parameter[parameter]  # scan parameter settings used
                    if len(scan_parameter_values) == 1:  # only analyze per scan step if there are more than one scan step
                        logging.warning('The file ' + str(input_file_hits) + ' has no different ' + str(parameter) + ' parameter values. Omit analysis.')
                    else:
                        logging.info('Analyze ' + input_file_hits + ' per scan parameter ' + parameter + ' for ' + str(len(scan_parameter_values)) + ' values from ' + str(np.amin(scan_parameter_values)) + ' to ' + str(np.amax(scan_parameter_values)))
                        event_numbers = analysis_utils.get_meta_data_at_scan_parameter(meta_data_array, parameter)['event_number']  # get the event numbers in meta_data where the scan parameter changes
                        parameter_ranges = np.column_stack((scan_parameter_values, analysis_utils.get_ranges_from_array(event_numbers)))
                        hit_table = in_hit_file_h5.root.Hits
                        analysis_utils.index_event_number(hit_table)
                        total_hits, total_hits_2, index = 0, 0, 0
                        chunk_size = max_chunk_size
                        # initialize the analysis and set settings
                        analyze_data = AnalyzeRawData()
                        analyze_data.create_cluster_size_hist = True
                        analyze_data.create_cluster_tot_hist = True
                        analyze_data.histogram.set_no_scan_parameter()  # one has to tell histogram the # of scan parameters for correct occupancy hist allocation
                        progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hit_table.shape[0], term_width=80)
                        progress_bar.start()
                        for parameter_index, parameter_range in enumerate(parameter_ranges):  # loop over the selected events
                            analyze_data.reset()  # resets the data of the last analysis
                            logging.debug('Analyze GDAC = ' + str(parameter_range[0]) + ' ' + str(int(float(float(parameter_index) / float(len(parameter_ranges)) * 100.0))) + '%')
                            start_event_number = parameter_range[1]
                            stop_event_number = parameter_range[2]
                            logging.debug('Data from events = [' + str(start_event_number) + ',' + str(stop_event_number) + '[')
                            actual_parameter_group = out_file_h5.create_group(parameter_goup, name=parameter + '_' + str(parameter_range[0]), title=parameter + '_' + str(parameter_range[0]))
                            # loop over the hits in the actual selected events with optimizations: variable chunk size, start word index given
                            readout_hit_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
                            for hits, index in analysis_utils.data_aligned_at_events(hit_table, start_event_number=start_event_number, stop_event_number=stop_event_number, start_index=index, chunk_size=chunk_size):
                                total_hits += hits.shape[0]
                                analyze_data.analyze_hits(hits)  # analyze the selected hits in chunks
                                readout_hit_len += hits.shape[0]
                                progress_bar.update(index)
                            chunk_size = int(1.05 * readout_hit_len) if int(1.05 * readout_hit_len) < max_chunk_size else max_chunk_size  # to increase the readout speed, estimated the number of hits for one read instruction
                            if chunk_size < 50:  # limit the lower chunk size, there can always be a crazy event with more than 20 hits
                                chunk_size = 50
                            # get occupancy hist
                            occupancy = analyze_data.histogram.get_occupancy()  # just check here if histogram is consistent

                            # store and plot cluster size hist
                            cluster_size_hist = analyze_data.clusterizer.get_cluster_size_hist()
                            cluster_size_hist_table = out_file_h5.create_carray(actual_parameter_group, name='HistClusterSize', title='Cluster Size Histogram', atom=tb.Atom.from_dtype(cluster_size_hist.dtype), shape=cluster_size_hist.shape, filters=filter_table)
                            cluster_size_hist_table[:] = cluster_size_hist
                            if output_pdf is not False:
                                plotting.plot_cluster_size(hist=cluster_size_hist, title='Cluster size (' + str(np.sum(cluster_size_hist)) + ' entries) for ' + parameter + ' = ' + str(scan_parameter_values[parameter_index]), filename=output_pdf)
                            if cluster_size_total is None:  # true if no data was appended to the array yet
                                cluster_size_total = cluster_size_hist
                            else:
                                cluster_size_total = np.vstack([cluster_size_total, cluster_size_hist])

                            total_hits_2 += np.sum(occupancy)
                        progress_bar.finish()
                        if total_hits != total_hits_2:
                            logging.warning('Analysis shows inconsistent number of hits. Check needed!')
                        logging.info('Analyzed %d hits!', total_hits)
            cluster_size_total_out = out_file_h5.create_carray(out_file_h5.root, name='AllHistClusterSize', title='All Cluster Size Histograms', atom=tb.Atom.from_dtype(cluster_size_total.dtype), shape=cluster_size_total.shape, filters=filter_table)
            cluster_size_total_out[:] = cluster_size_total