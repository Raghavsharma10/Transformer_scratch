def analyze_event_rate(scan_base, combine_n_readouts=1000, time_line_absolute=True, output_pdf=None, output_file=None):
    ''' Determines the number of events as a function of time. Therefore the data of a fixed number of read outs are combined ('combine_n_readouts'). The number of events is taken from the meta data info
    and stored into a pdf file.

    Parameters
    ----------
    scan_base: list of str
        scan base names (e.g.:  ['//data//SCC_50_fei4_self_trigger_scan_390', ]
    combine_n_readouts: int
        the number of read outs to combine (e.g. 1000)
    time_line_absolute: bool
        if true the analysis uses absolute time stamps
    output_pdf: PdfPages
        PdfPages file object, if none the plot is printed to screen
    '''
    time_stamp = []
    rate = []

    start_time_set = False

    for data_file in scan_base:
        with tb.open_file(data_file + '_interpreted.h5', mode="r") as in_file_h5:
            meta_data_array = in_file_h5.root.meta_data[:]
            parameter_ranges = np.column_stack((analysis_utils.get_ranges_from_array(meta_data_array['timestamp_start'][::combine_n_readouts]), analysis_utils.get_ranges_from_array(meta_data_array['event_number'][::combine_n_readouts])))

            if time_line_absolute:
                time_stamp.extend(parameter_ranges[:-1, 0])
            else:
                if not start_time_set:
                    start_time = parameter_ranges[0, 0]
                    start_time_set = True
                time_stamp.extend((parameter_ranges[:-1, 0] - start_time) / 60.0)
            rate.extend((parameter_ranges[:-1, 3] - parameter_ranges[:-1, 2]) / (parameter_ranges[:-1, 1] - parameter_ranges[:-1, 0]))  # d#Events / dt
    if time_line_absolute:
        plotting.plot_scatter_time(time_stamp, rate, title='Event rate [Hz]', marker_style='o', filename=output_pdf)
    else:
        plotting.plot_scatter(time_stamp, rate, title='Events per time', x_label='Progressed time [min.]', y_label='Events rate [Hz]', marker_style='o', filename=output_pdf)
    if output_file:
        with tb.open_file(output_file, mode="a") as out_file_h5:
            rec_array = np.array(zip(time_stamp, rate), dtype=[('time_stamp', float), ('rate', float)]).view(np.recarray)
            try:
                rate_table = out_file_h5.create_table(out_file_h5.root, name='Eventrate', description=rec_array, title='Event rate', filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
                rate_table[:] = rec_array
            except tb.exceptions.NodeError:
                logging.warning(output_file + ' has already a Eventrate note, do not overwrite existing.')
    return time_stamp, rate