def create_inputs_to_reference(job_data, input_files, input_directories):
    """
    Creates a dictionary with the summarized information in job_data, input_files and input_directories

    :param job_data: The job data specifying input parameters other than files and directories.
    :param input_files: A dictionary describing the input files.
    :param input_directories: A dictionary describing the input directories.
    :return: A summarized dictionary containing information about all given inputs.
    """

    return {**deepcopy(job_data), **deepcopy(input_files), **deepcopy(input_directories)}