def cwl_output_files(cwl_data, inputs_to_reference, output_dir=None):
    """
    Returns a dictionary containing information about the output files given in cwl_data.

    :param cwl_data: The cwl data from where to extract the output file information.
    :param inputs_to_reference: Inputs which are used to resolve input references.
    :param output_dir: Path to the directory where output files are expected.
    :return: A dictionary containing information about every output file.
    """
    results = {}

    for key, val in cwl_data['outputs'].items():
        cwl_type = parse_cwl_type(val['type'])
        (is_optional, is_array, cwl_type) = itemgetter('isOptional', 'isArray', 'type')(cwl_type)

        if not cwl_type == 'File':
            continue

        result = {
            'isOptional': is_optional,
            'path': None,
            'size': None,
            'debugInfo': None
        }

        glob_path = os.path.expanduser(val['outputBinding']['glob'])
        if output_dir and not os.path.isabs(glob_path):
            glob_path = os.path.join(os.path.expanduser(output_dir), glob_path)

        glob_path = resolve_input_references(glob_path, inputs_to_reference)
        matches = glob(glob_path)
        try:
            if len(matches) != 1:
                raise FileError('glob path "{}" does not match exactly one file'.format(glob_path))

            file_path = matches[0]
            result['path'] = file_path

            if not os.path.isfile(file_path):
                raise FileError('path is not a file')

            result['size'] = os.path.getsize(file_path) / (1024 * 1024)
        except:
            result['debugInfo'] = exception_format()

        results[key] = result

    return results