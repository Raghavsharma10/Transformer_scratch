def generate_contour_data(pid):
    """
    Main function for this program.

    This will read in sensitivity_curves and binary parameters; calculate snrs
    with a matched filtering approach; and then read the contour data out to a file.

    Args:
        pid (obj or dict): GenInput class or dictionary containing all of the input information for
            the generation. See BOWIE documentation and example notebooks for usage of
            this class.

    """
    # check if pid is  dicionary or GenInput class
    # if GenInput, change to dictionary
    if isinstance(pid, GenInput):
        pid = pid.return_dict()

    begin_time = time.time()

    WORKING_DIRECTORY = '.'
    if 'WORKING_DIRECTORY' not in pid['general'].keys():
        pid['general']['WORKING_DIRECTORY'] = WORKING_DIRECTORY

    # Generate the contour data.
    running_process = GenProcess(**{**pid, **pid['generate_info']})
    running_process.set_parameters()
    running_process.run_snr()

    # Read out
    file_out = FileReadOut(running_process.xvals, running_process.yvals,
                           running_process.final_dict,
                           **{**pid['general'], **pid['generate_info'], **pid['output_info']})

    print('outputing file:', pid['general']['WORKING_DIRECTORY'] + '/'
          + pid['output_info']['output_file_name'])
    getattr(file_out, file_out.output_file_type + '_read_out')()

    print(time.time()-begin_time)
    return