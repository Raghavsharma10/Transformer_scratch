def get_rate_normalization(hit_file, parameter, reference='event', cluster_file=None, plot=False, chunk_size=500000):
    ''' Takes different hit files (hit_files), extracts the number of events or the scan time (reference) per scan parameter (parameter)
    and returns an array with a normalization factor. This normalization factor has the length of the number of different parameters.
    If a cluster_file is specified also the number of cluster per event are used to create the normalization factor.

    Parameters
    ----------
    hit_files : string
    parameter : string
    reference : string
    plot : bool

    Returns
    -------
    numpy.ndarray
    '''

    logging.info('Calculate the rate normalization')
    with tb.open_file(hit_file, mode="r+") as in_hit_file_h5:  # open the hit file
        meta_data = in_hit_file_h5.root.meta_data[:]
        scan_parameter = get_scan_parameter(meta_data)[parameter]
        event_numbers = get_meta_data_at_scan_parameter(meta_data, parameter)['event_number']  # get the event numbers in meta_data where the scan parameter changes
        event_range = get_ranges_from_array(event_numbers)
        normalization_rate = []
        normalization_multiplicity = []
        try:
            event_range[-1, 1] = in_hit_file_h5.root.Hits[-1]['event_number'] + 1
        except tb.NoSuchNodeError:
            logging.error('Cannot find hits table')
            return

        # calculate rate normalization from the event rate for triggered data / measurement time for self triggered data for each scan parameter
        if reference == 'event':
            n_events = event_range[:, 1] - event_range[:, 0]  # number of events for every parameter setting
            normalization_rate.extend(n_events)
        elif reference == 'time':
            time_start = get_meta_data_at_scan_parameter(meta_data, parameter)['timestamp_start']
            time_spend = np.diff(time_start)
            time_spend = np.append(time_spend, meta_data[-1]['timestamp_stop'] - time_start[-1])  # TODO: needs check, add last missing entry
            normalization_rate.extend(time_spend)
        else:
            raise NotImplementedError('The normalization reference ' + reference + ' is not implemented')

        if cluster_file:  # calculate the rate normalization from the mean number of hits per event per scan parameter, needed for beam data since a beam since the multiplicity is rarely constant
            cluster_table = in_hit_file_h5.root.Cluster
            index_event_number(cluster_table)
            index = 0  # index where to start the read out, 0 at the beginning, increased during looping, variable for read speed up
            best_chunk_size = chunk_size  # variable for read speed up
            total_cluster = 0
            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=cluster_table.shape[0], term_width=80)
            progress_bar.start()
            for start_event, stop_event in event_range:  # loop over the selected events
                readout_cluster_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
                n_cluster_per_event = None
                for clusters, index in data_aligned_at_events(cluster_table, start_event_number=start_event, stop_event_number=stop_event, start_index=index, chunk_size=best_chunk_size):
                    if n_cluster_per_event is None:
                        n_cluster_per_event = analysis_utils.get_n_cluster_in_events(clusters['event_number'])[:, 1]  # array with the number of cluster per event, cluster per event are at least 1
                    else:
                        n_cluster_per_event = np.append(n_cluster_per_event, analysis_utils.get_n_cluster_in_events(clusters['event_number'])[:, 1])
                    readout_cluster_len += clusters.shape[0]
                    total_cluster += clusters.shape[0]
                    progress_bar.update(index)
                best_chunk_size = int(1.5 * readout_cluster_len) if int(1.05 * readout_cluster_len) < chunk_size else chunk_size  # to increase the readout speed, estimated the number of hits for one read instruction
                normalization_multiplicity.append(np.mean(n_cluster_per_event))
            progress_bar.finish()
            if total_cluster != cluster_table.shape[0]:
                logging.warning('Analysis shows inconsistent number of cluster (%d != %d). Check needed!', total_cluster, cluster_table.shape[0])

    if plot:
        x = scan_parameter
        if reference == 'event':
            plotting.plot_scatter(x, normalization_rate, title='Events per ' + parameter + ' setting', x_label=parameter, y_label='# events', log_x=True, filename=os.path.splitext(hit_file)[0] + '_n_event_normalization.pdf')
        elif reference == 'time':
            plotting.plot_scatter(x, normalization_rate, title='Measuring time per GDAC setting', x_label=parameter, y_label='time [s]', log_x=True, filename=os.path.splitext(hit_file)[0] + '_time_normalization.pdf')
        if cluster_file:
            plotting.plot_scatter(x, normalization_multiplicity, title='Mean number of particles per event', x_label=parameter, y_label='number of hits per event', log_x=True, filename=os.path.splitext(hit_file)[0] + '_n_particles_normalization.pdf')
    if cluster_file:
        normalization_rate = np.array(normalization_rate)
        normalization_multiplicity = np.array(normalization_multiplicity)
        return np.amax(normalization_rate * normalization_multiplicity).astype('f16') / (normalization_rate * normalization_multiplicity)
    return np.amax(np.array(normalization_rate)).astype('f16') / np.array(normalization_rate)