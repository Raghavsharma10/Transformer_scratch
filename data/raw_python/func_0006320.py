def analyse_n_cluster_per_event(scan_base, include_no_cluster=False, time_line_absolute=True, combine_n_readouts=1000, chunk_size=10000000, plot_n_cluster_hists=False, output_pdf=None, output_file=None):
    ''' Determines the number of cluster per event as a function of time. Therefore the data of a fixed number of read outs are combined ('combine_n_readouts').

    Parameters
    ----------
    scan_base: list of str
        scan base names (e.g.:  ['//data//SCC_50_fei4_self_trigger_scan_390', ]
    include_no_cluster: bool
        Set to true to also consider all events without any hit.
    combine_n_readouts: int
        the number of read outs to combine (e.g. 1000)
    max_chunk_size: int
        the maximum chunk size used during read, if too big memory error occurs, if too small analysis takes longer
    output_pdf: PdfPages
        PdfPages file object, if none the plot is printed to screen
    '''

    time_stamp = []
    n_cluster = []

    start_time_set = False

    for data_file in scan_base:
        with tb.open_file(data_file + '_interpreted.h5', mode="r+") as in_cluster_file_h5:
            # get data and data pointer
            meta_data_array = in_cluster_file_h5.root.meta_data[:]
            cluster_table = in_cluster_file_h5.root.Cluster

            # determine the event ranges to analyze (timestamp_start, start_event_number, stop_event_number)
            parameter_ranges = np.column_stack((analysis_utils.get_ranges_from_array(meta_data_array['timestamp_start'][::combine_n_readouts]), analysis_utils.get_ranges_from_array(meta_data_array['event_number'][::combine_n_readouts])))

            # create a event_numer index (important for speed)
            analysis_utils.index_event_number(cluster_table)

            # initialize the analysis and set settings
            analyze_data = AnalyzeRawData()
            analyze_data.create_tot_hist = False
            analyze_data.create_bcid_hist = False

            # variables for read speed up
            index = 0  # index where to start the read out, 0 at the beginning, increased during looping
            best_chunk_size = chunk_size

            total_cluster = cluster_table.shape[0]

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=total_cluster, term_width=80)
            progress_bar.start()

            # loop over the selected events
            for parameter_index, parameter_range in enumerate(parameter_ranges):
                logging.debug('Analyze time stamp ' + str(parameter_range[0]) + ' and data from events = [' + str(parameter_range[2]) + ',' + str(parameter_range[3]) + '[ ' + str(int(float(float(parameter_index) / float(len(parameter_ranges)) * 100.0))) + '%')
                analyze_data.reset()  # resets the data of the last analysis

                # loop over the cluster in the actual selected events with optimizations: determine best chunk size, start word index given
                readout_cluster_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
                hist = None
                for clusters, index in analysis_utils.data_aligned_at_events(cluster_table, start_event_number=parameter_range[2], stop_event_number=parameter_range[3], start_index=index, chunk_size=best_chunk_size):
                    n_cluster_per_event = analysis_utils.get_n_cluster_in_events(clusters['event_number'])[:, 1]  # array with the number of cluster per event, cluster per event are at least 1
                    if hist is None:
                        hist = np.histogram(n_cluster_per_event, bins=10, range=(0, 10))[0]
                    else:
                        hist = np.add(hist, np.histogram(n_cluster_per_event, bins=10, range=(0, 10))[0])
                    if include_no_cluster and parameter_range[3] is not None:  # happend for the last readout
                        hist[0] = (parameter_range[3] - parameter_range[2]) - len(n_cluster_per_event)  # add the events without any cluster
                    readout_cluster_len += clusters.shape[0]
                    total_cluster -= len(clusters)
                    progress_bar.update(index)
                best_chunk_size = int(1.5 * readout_cluster_len) if int(1.05 * readout_cluster_len) < chunk_size else chunk_size  # to increase the readout speed, estimated the number of hits for one read instruction

                if plot_n_cluster_hists:
                    plotting.plot_1d_hist(hist, title='Number of cluster per event at ' + str(parameter_range[0]), x_axis_title='Number of cluster', y_axis_title='#', log_y=True, filename=output_pdf)
                hist = hist.astype('f4') / np.sum(hist)  # calculate fraction from total numbers

                if time_line_absolute:
                    time_stamp.append(parameter_range[0])
                else:
                    if not start_time_set:
                        start_time = parameter_ranges[0, 0]
                        start_time_set = True
                    time_stamp.append((parameter_range[0] - start_time) / 60.0)
                n_cluster.append(hist)
            progress_bar.finish()
            if total_cluster != 0:
                logging.warning('Not all clusters were selected during analysis. Analysis is therefore not exact')

    if time_line_absolute:
        plotting.plot_scatter_time(time_stamp, n_cluster, title='Number of cluster per event as a function of time', marker_style='o', filename=output_pdf, legend=('0 cluster', '1 cluster', '2 cluster', '3 cluster') if include_no_cluster else ('0 cluster not plotted', '1 cluster', '2 cluster', '3 cluster'))
    else:
        plotting.plot_scatter(time_stamp, n_cluster, title='Number of cluster per event as a function of time', x_label='time [min.]', marker_style='o', filename=output_pdf, legend=('0 cluster', '1 cluster', '2 cluster', '3 cluster') if include_no_cluster else ('0 cluster not plotted', '1 cluster', '2 cluster', '3 cluster'))
    if output_file:
        with tb.open_file(output_file, mode="a") as out_file_h5:
            cluster_array = np.array(n_cluster)
            rec_array = np.array(zip(time_stamp, cluster_array[:, 0], cluster_array[:, 1], cluster_array[:, 2], cluster_array[:, 3], cluster_array[:, 4], cluster_array[:, 5]), dtype=[('time_stamp', float), ('cluster_0', float), ('cluster_1', float), ('cluster_2', float), ('cluster_3', float), ('cluster_4', float), ('cluster_5', float)]).view(np.recarray)
            try:
                n_cluster_table = out_file_h5.create_table(out_file_h5.root, name='n_cluster', description=rec_array, title='Cluster per event', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                n_cluster_table[:] = rec_array
            except tb.exceptions.NodeError:
                logging.warning(output_file + ' has already a Beamspot note, do not overwrite existing.')
    return time_stamp, n_cluster