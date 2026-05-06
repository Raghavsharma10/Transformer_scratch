def parse_cwl_type(cwl_type_string):
    """
    Parses cwl type information from a cwl type string.

    Examples:

    - "File[]" -> {'type': 'File', 'isArray': True, 'isOptional': False}
    - "int?" -> {'type': 'int', 'isArray': False, 'isOptional': True}

    :param cwl_type_string: The cwl type string to extract information from
    :return: A dictionary containing information about the parsed cwl type string
    """

    is_optional = cwl_type_string.endswith('?')
    if is_optional:
        cwl_type_string = cwl_type_string[:-1]

    is_array = cwl_type_string.endswith('[]')
    if is_array:
        cwl_type_string = cwl_type_string[:-2]

    return {'type': cwl_type_string, 'isArray': is_array, 'isOptional': is_optional}