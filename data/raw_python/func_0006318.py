def analyze_beam_spot(scan_base, combine_n_readouts=1000, chunk_size=10000000, plot_occupancy_hists=False, output_pdf=None, output_file=None):
    ''' Determines the mean x and y beam spot position as a function of time. Therefore the data of a fixed number of read outs are combined ('combine_n_readouts'). The occupancy is determined
    for the given combined events and stored into a pdf file. At the end the beam x and y is plotted into a scatter plot with absolute positions in um.

     Parameters
    ----------
    scan_base: list of str
        scan base names (e.g.:  ['//data//SCC_50_fei4_self_trigger_scan_390', ]
    combine_n_readouts: int
        the number of read outs to combine (e.g. 1000)
    max_chunk_size: int
        the maximum chunk size used during read, if too big memory error occurs, if too small analysis takes longer
    output_pdf: PdfPages
        PdfPages file object, if none the plot is printed to screen
    '''
    time_stamp = []
    x = []
    y = []

    for data_file in scan_base:
        with tb.open_file(data_file + '_interpreted.h5', mode="r+") as in_hit_file_h5:
            # get data and data pointer
            meta_data_array = in_hit_file_h5.root.meta_data[:]
            hit_table = in_hit_file_h5.root.Hits

            # determine the event ranges to analyze (timestamp_start, start_event_number, stop_event_number)
            parameter_ranges = np.column_stack((analysis_utils.get_ranges_from_array(meta_data_array['timestamp_start'][::combine_n_readouts]), analysis_utils.get_ranges_from_array(meta_data_array['event_number'][::combine_n_readouts])))

            # create a event_numer index (important)
            analysis_utils.index_event_number(hit_table)

            # initialize the analysis and set settings
            analyze_data = AnalyzeRawData()
            analyze_data.create_tot_hist = False
            analyze_data.create_bcid_hist = False
            analyze_data.histogram.set_no_scan_parameter()

            # variables for read speed up
            index = 0  # index where to start the read out, 0 at the beginning, increased during looping
            best_chunk_size = chunk_size

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=hit_table.shape[0], term_width=80)
            progress_bar.start()

            # loop over the selected events
            for parameter_index, parameter_range in enumerate(parameter_ranges):
                logging.debug('Analyze time stamp ' + str(parameter_range[0]) + ' and data from events = [' + str(parameter_range[2]) + ',' + str(parameter_range[3]) + '[ ' + str(int(float(float(parameter_index) / float(len(parameter_ranges)) * 100.0))) + '%')
                analyze_data.reset()  # resets the data of the last analysis

                # loop over the hits in the actual selected events with optimizations: determine best chunk size, start word index given
                readout_hit_len = 0  # variable to calculate a optimal chunk size value from the number of hits for speed up
                for hits, index in analysis_utils.data_aligned_at_events(hit_table, start_event_number=parameter_range[2], stop_event_number=parameter_range[3], start_index=index, chunk_size=best_chunk_size):
                    analyze_data.analyze_hits(hits)  # analyze the selected hits in chunks
                    readout_hit_len += hits.shape[0]
                    progress_bar.update(index)
                best_chunk_size = int(1.5 * readout_hit_len) if int(1.05 * readout_hit_len) < chunk_size else chunk_size  # to increase the readout speed, estimated the number of hits for one read instruction

                # get and store results
                occupancy_array = analyze_data.histogram.get_occupancy()
                projection_x = np.sum(occupancy_array, axis=0).ravel()
                projection_y = np.sum(occupancy_array, axis=1).ravel()
                x.append(analysis_utils.get_mean_from_histogram(projection_x, bin_positions=range(0, 80)))
                y.append(analysis_utils.get_mean_from_histogram(projection_y, bin_positions=range(0, 336)))
                time_stamp.append(parameter_range[0])
                if plot_occupancy_hists:
                    plotting.plot_occupancy(occupancy_array[:, :, 0], title='Occupancy for events between ' + time.strftime('%H:%M:%S', time.localtime(parameter_range[0])) + ' and ' + time.strftime('%H:%M:%S', time.localtime(parameter_range[1])), filename=output_pdf)
            progress_bar.finish()
    plotting.plot_scatter([i * 250 for i in x], [i * 50 for i in y], title='Mean beam position', x_label='x [um]', y_label='y [um]', marker_style='-o', filename=output_pdf)
    if output_file:
        with tb.open_file(output_file, mode="a") as out_file_h5:
            rec_array = np.array(zip(time_stamp, x, y), dtype=[('time_stamp', float), ('x', float), ('y', float)])
            try:
                beam_spot_table = out_file_h5.create_table(out_file_h5.root, name='Beamspot', description=rec_array, title='Beam spot position', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                beam_spot_table[:] = rec_array
            except tb.exceptions.NodeError:
                logging.warning(output_file + ' has already a Beamspot note, do not overwrite existing.')
    return time_stamp, x, y