def create_hitor_calibration(output_filename, plot_pixel_calibrations=False):
    '''Generating HitOr calibration file (_calibration.h5) from raw data file and plotting of calibration data.

    Parameters
    ----------
    output_filename : string
        Input raw data file name.
    plot_pixel_calibrations : bool, iterable
        If True, genearating additional pixel calibration plots. If list of column and row tuples (from 1 to 80 / 336), print selected pixels.

    Returns
    -------
    nothing
    '''
    logging.info('Analyze HitOR calibration data and plot results of %s', output_filename)

    with AnalyzeRawData(raw_data_file=output_filename, create_pdf=True) as analyze_raw_data:  # Interpret the raw data file
        analyze_raw_data.create_occupancy_hist = False  # too many scan parameters to do in ram histogramming
        analyze_raw_data.create_hit_table = True
        analyze_raw_data.create_tdc_hist = True
        analyze_raw_data.align_at_tdc = True  # align events at TDC words, first word of event has to be a tdc word
        analyze_raw_data.interpret_word_table()
        analyze_raw_data.interpreter.print_summary()
        analyze_raw_data.plot_histograms()

        n_injections = analyze_raw_data.n_injections  # use later
        meta_data = analyze_raw_data.out_file_h5.root.meta_data[:]
        scan_parameters_dict = get_scan_parameter(meta_data)
        inner_loop_parameter_values = scan_parameters_dict[next(reversed(scan_parameters_dict))]  # inner loop parameter name is unknown
        scan_parameter_names = scan_parameters_dict.keys()
#         col_row_combinations = get_unique_scan_parameter_combinations(analyze_raw_data.out_file_h5.root.meta_data[:], scan_parameters=('column', 'row'), scan_parameter_columns_only=True)

        meta_data_table_at_scan_parameter = get_unique_scan_parameter_combinations(meta_data, scan_parameters=scan_parameter_names)
        scan_parameter_values = get_scan_parameters_table_from_meta_data(meta_data_table_at_scan_parameter, scan_parameter_names)
        event_number_ranges = get_ranges_from_array(meta_data_table_at_scan_parameter['event_number'])
        event_ranges_per_parameter = np.column_stack((scan_parameter_values, event_number_ranges))
        if analyze_raw_data.out_file_h5.root.Hits.nrows == 0:
            raise AnalysisError("Found no hits.")
        hits = analyze_raw_data.out_file_h5.root.Hits[:]
        event_numbers = hits['event_number'].copy()  # create contigous array, otherwise np.searchsorted too slow, http://stackoverflow.com/questions/15139299/performance-of-numpy-searchsorted-is-poor-on-structured-arrays

        output_filename = os.path.splitext(output_filename)[0]
        with tb.open_file(output_filename + "_calibration.h5", mode="w") as calibration_data_file:
            logging.info('Create calibration')
            calibration_data = np.full(shape=(80, 336, len(inner_loop_parameter_values), 4), fill_value=np.nan, dtype='f4')  # result of the calibration is a histogram with col_index, row_index, plsrDAC value, mean discrete tot, rms discrete tot, mean tot from TDC, rms tot from TDC

            progress_bar = progressbar.ProgressBar(widgets=['', progressbar.Percentage(), ' ', progressbar.Bar(marker='*', left='|', right='|'), ' ', progressbar.AdaptiveETA()], maxval=len(event_ranges_per_parameter), term_width=80)
            progress_bar.start()

            for index, (actual_scan_parameter_values, event_start, event_stop) in enumerate(event_ranges_per_parameter):
                if event_stop is None:  # happens for the last chunk
                    event_stop = hits[-1]['event_number'] + 1
                array_index = np.searchsorted(event_numbers, np.array([event_start, event_stop]))
                actual_hits = hits[array_index[0]:array_index[1]]
                for item_index, item in enumerate(scan_parameter_names):
                    if item == "column":
                        actual_col = actual_scan_parameter_values[item_index]
                    elif item == "row":
                        actual_row = actual_scan_parameter_values[item_index]
                    elif item == "PlsrDAC":
                        plser_dac = actual_scan_parameter_values[item_index]
                    else:
                        raise ValueError("Unknown scan parameter %s" % item)

                # Only pixel of actual column/row should be in the actual data chunk but since FIFO is not cleared for each scan step due to speed reasons and there might be noisy pixels this is not always the case
                n_wrong_pixel = np.count_nonzero(np.logical_or(actual_hits['column'] != actual_col, actual_hits['row'] != actual_row))
                if n_wrong_pixel != 0:
                    logging.warning('%d hit(s) from other pixels for scan parameters %s', n_wrong_pixel, ', '.join(['%s=%s' % (name, value) for (name, value) in zip(scan_parameter_names, actual_scan_parameter_values)]))

                actual_hits = actual_hits[np.logical_and(actual_hits['column'] == actual_col, actual_hits['row'] == actual_row)]  # Only take data from selected pixel
                actual_tdc_hits = actual_hits[(actual_hits['event_status'] & 0b0000111110011100) == 0b0000000100000000]  # only take hits from good events (one TDC word only, no error)
                actual_tot_hits = actual_hits[(actual_hits['event_status'] & 0b0000100010011100) == 0b0000000000000000]  # only take hits from good events for tot
                tot, tdc = actual_tot_hits['tot'], actual_tdc_hits['TDC']

                if tdc.shape[0] < n_injections:
                    logging.info('%d of %d expected TDC hits for scan parameters %s', tdc.shape[0], n_injections, ', '.join(['%s=%s' % (name, value) for (name, value) in zip(scan_parameter_names, actual_scan_parameter_values)]))
                if tot.shape[0] < n_injections:
                    logging.info('%d of %d expected hits for scan parameters %s', tot.shape[0], n_injections, ', '.join(['%s=%s' % (name, value) for (name, value) in zip(scan_parameter_names, actual_scan_parameter_values)]))

                inner_loop_scan_parameter_index = np.where(plser_dac == inner_loop_parameter_values)[0][0]  # translate the scan parameter value to an index for the result histogram
                # numpy mean and std return nan if array is empty
                calibration_data[actual_col - 1, actual_row - 1, inner_loop_scan_parameter_index, 0] = np.mean(tot)
                calibration_data[actual_col - 1, actual_row - 1, inner_loop_scan_parameter_index, 1] = np.mean(tdc)
                calibration_data[actual_col - 1, actual_row - 1, inner_loop_scan_parameter_index, 2] = np.std(tot)
                calibration_data[actual_col - 1, actual_row - 1, inner_loop_scan_parameter_index, 3] = np.std(tdc)

                progress_bar.update(index)
            progress_bar.finish()

            calibration_data_out = calibration_data_file.create_carray(calibration_data_file.root, name='HitOrCalibration', title='Hit OR calibration data', atom=tb.Atom.from_dtype(calibration_data.dtype), shape=calibration_data.shape, filters=tb.Filters(complib='blosc', complevel=5, fletcher32=False))
            calibration_data_out[:] = calibration_data
            calibration_data_out.attrs.dimensions = scan_parameter_names
            calibration_data_out.attrs.scan_parameter_values = inner_loop_parameter_values
            calibration_data_out.flush()
#             with PdfPages(output_filename + "_calibration.pdf") as output_pdf:
            plot_scurves(calibration_data[:, :, :, 0], inner_loop_parameter_values, "ToT calibration", "ToT", 15, "Charge [PlsrDAC]", filename=analyze_raw_data.output_pdf)
            plot_scurves(calibration_data[:, :, :, 1], inner_loop_parameter_values, "TDC calibration", "TDC [ns]", None, "Charge [PlsrDAC]", filename=analyze_raw_data.output_pdf)
            tot_mean_all_pix = np.nanmean(calibration_data[:, :, :, 0], axis=(0, 1))
            tot_error_all_pix = np.nanstd(calibration_data[:, :, :, 0], axis=(0, 1))
            tdc_mean_all_pix = np.nanmean(calibration_data[:, :, :, 1], axis=(0, 1))
            tdc_error_all_pix = np.nanstd(calibration_data[:, :, :, 1], axis=(0, 1))
            plot_tot_tdc_calibration(scan_parameters=inner_loop_parameter_values, tot_mean=tot_mean_all_pix, tot_error=tot_error_all_pix, tdc_mean=tdc_mean_all_pix, tdc_error=tdc_error_all_pix, filename=analyze_raw_data.output_pdf, title="Mean charge calibration of %d pixel(s)" % np.count_nonzero(~np.all(np.isnan(calibration_data[:, :, :, 0]), axis=2)))
            # plotting individual pixels
            if plot_pixel_calibrations is True:
                # selecting pixels with non-nan entries
                col_row_non_nan = np.nonzero(~np.all(np.isnan(calibration_data[:, :, :, 0]), axis=2))
                plot_pixel_calibrations = np.dstack(col_row_non_nan)[0]
            elif plot_pixel_calibrations is False:
                plot_pixel_calibrations = np.array([], dtype=np.int)
            else:  # assuming list of column / row tuples
                plot_pixel_calibrations = np.array(plot_pixel_calibrations) - 1
            # generate index array
            pixel_indices = np.arange(plot_pixel_calibrations.shape[0])
            plot_n_pixels = 10  # number of pixels at the beginning, center and end of the array
            np.random.seed(0)
            # select random pixels
            if pixel_indices.size - 2 * plot_n_pixels >= 0:
                random_pixel_indices = np.sort(np.random.choice(pixel_indices[plot_n_pixels:-plot_n_pixels], min(plot_n_pixels, pixel_indices.size - 2 * plot_n_pixels), replace=False))
            else:
                random_pixel_indices = np.array([], dtype=np.int)
            selected_pixel_indices = np.unique(np.hstack([pixel_indices[:plot_n_pixels], random_pixel_indices, pixel_indices[-plot_n_pixels:]]))
            # plotting individual pixels
            for (column, row) in plot_pixel_calibrations[selected_pixel_indices]:
                logging.info("Plotting charge calibration for pixel column " + str(column + 1) + " / row " + str(row + 1))
                tot_mean_single_pix = calibration_data[column, row, :, 0]
                tot_std_single_pix = calibration_data[column, row, :, 2]
                tdc_mean_single_pix = calibration_data[column, row, :, 1]
                tdc_std_single_pix = calibration_data[column, row, :, 3]
                plot_tot_tdc_calibration(scan_parameters=inner_loop_parameter_values, tot_mean=tot_mean_single_pix, tot_error=tot_std_single_pix, tdc_mean=tdc_mean_single_pix, tdc_error=tdc_std_single_pix, filename=analyze_raw_data.output_pdf, title="Charge calibration for pixel column " + str(column + 1) + " / row " + str(row + 1))