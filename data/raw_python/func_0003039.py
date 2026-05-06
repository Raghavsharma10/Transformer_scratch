def resolve_input_reference(reference, inputs_to_reference):
    """
    Replaces a given input_reference by a string extracted from inputs_to_reference.

    :param reference: The input reference to resolve.
    :param inputs_to_reference: A dictionary containing information about the given inputs.

    :raise InvalidInputReference: If the given input reference could not be resolved.

    :return: A string which is the resolved input reference.
    """
    if not reference.startswith('{}inputs.'.format(INPUT_REFERENCE_START)):
        raise InvalidInputReference('An input reference must have the following form'
                                    '"$(inputs.<input_name>[.<attribute>]".\n'
                                    'The invalid reference is: "{}"'.format(reference))
    # remove "$(inputs." and ")"
    reference = reference[2:-1]
    parts = split_all(reference, ATTRIBUTE_SEPARATOR_SYMBOLS)

    if len(parts) < 2:
        raise InvalidInputReference('InputReference should at least contain "$(inputs.identifier)". The following input'
                                    'reference does not comply with it:\n{}'.format(reference))
    elif parts[0] != "inputs":
        raise InvalidInputReference('InputReference should at least contain "$(inputs.identifier)". The following input'
                                    ' reference does not comply with it:\n$({})'.format(reference))
    else:
        input_identifier = parts[1]
        input_to_reference = inputs_to_reference.get(input_identifier)
        if input_to_reference is None:
            raise InvalidInputReference('Input identifier "{}" not found in inputs, but needed in input reference:\n{}'
                                        .format(input_identifier, reference))
        elif isinstance(input_to_reference, dict):
            if 'files' in input_to_reference:
                return _resolve_file(parts[2:], input_to_reference, input_identifier, reference)
            elif 'directories' in input_to_reference:
                return _resolve_directory(parts[2:], input_to_reference, input_identifier, reference)
            else:
                raise InvalidInputReference('Unknown input type for input identifier "{}"'.format(input_identifier))
        else:
            if len(parts) > 2:
                raise InvalidInputReference('Attribute "{}" of input reference "{}" could not be resolved'
                                            .format(parts[2], reference))
            else:
                return parts[1]