def cwl_input_directories(cwl_data, job_data, input_dir=None):
    """
    Searches for Directories and in the cwl data and produces a dictionary containing input file information.

    :param cwl_data: The cwl data as dictionary
    :param job_data: The job data as dictionary
    :param input_dir: TODO
    :return: Returns the a dictionary containing information about input files.
             The keys of this dictionary are the input/output identifiers of the files specified in the cwl description.
             The corresponding value is a dictionary again with the following keys and values:
             - 'isOptional': A bool indicating whether this input directory is optional
             - 'isArray': A bool indicating whether this could be a list of directories
             - 'files': A list of input file descriptions

             A input file description is a dictionary containing the following information
             - 'path': The path to the specified directory
             - 'debugInfo': A field to possibly provide debug information
    """

    results = {}

    for input_identifier, input_data in cwl_data['inputs'].items():
        cwl_type = parse_cwl_type(input_data['type'])

        (is_optional, is_array, cwl_type) = itemgetter('isOptional', 'isArray', 'type')(cwl_type)

        if cwl_type == 'Directory':
            result = {
                'isOptional': is_optional,
                'isArray': is_array,
                'directories': None
            }

            if input_identifier in job_data:
                arg = job_data[input_identifier]

                if is_array:
                    result['directories'] = [_input_directory_description(input_identifier, i, input_dir) for i in arg]
                else:
                    result['directories'] = [_input_directory_description(input_identifier, arg, input_dir)]

            results[input_identifier] = result

    return results