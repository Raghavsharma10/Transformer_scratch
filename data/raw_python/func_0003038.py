def _resolve_directory(attributes, input_directory, input_identifier, input_reference):
    """
    Returns the attributes in demand of the input directory.

    :param attributes: A list of attributes to get from the input directory.
    :param input_directory: The directory from which to get the attributes.
    :param input_identifier: The input identifier of the given directory.
    :param input_reference: The reference string
    :return: The attribute in demand
    """
    if input_directory['isArray']:
        raise InvalidInputReference('Input References to Arrays of input directories are currently not supported.\n'
                                    'input directory "{}" is an array of directories and can not be resolved for input'
                                    'references:\n{}'.format(input_identifier, input_reference))
    single_directory = input_directory['directories'][0]

    try:
        return _get_dict_element(single_directory, attributes)
    except KeyError:
        raise InvalidInputReference('Could not get attributes "{}" from input directory "{}", needed in input'
                                    'reference:\n{}'.format(attributes, input_identifier, input_reference))