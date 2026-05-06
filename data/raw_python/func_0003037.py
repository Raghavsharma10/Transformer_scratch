def _resolve_file(attributes, input_file, input_identifier, input_reference):
    """
    Returns the attributes in demand of the input file.

    :param attributes: A list of attributes to get from the input_file.
    :param input_file: The file from which to get the attributes.
    :param input_identifier: The input identifier of the given file.
    :param input_reference: The reference string
    :return: The attribute in demand
    """
    if input_file['isArray']:
        raise InvalidInputReference('Input References to Arrays of input files are currently not supported.\n'
                                    '"{}" is an array of files and can not be resolved for input references:'
                                    '\n{}'.format(input_identifier, input_reference))
    single_file = input_file['files'][0]

    try:
        return _get_dict_element(single_file, attributes)
    except KeyError:
        raise InvalidInputReference('Could not get attributes "{}" from input file "{}", needed in input reference:'
                                    '\n{}'.format(attributes, input_identifier, input_reference))