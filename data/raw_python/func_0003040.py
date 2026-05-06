def resolve_input_references(to_resolve, inputs_to_reference):
    """
    Resolves input references given in the string to_resolve by using the inputs_to_reference.

    See http://www.commonwl.org/user_guide/06-params/index.html for more information.

    Example:
    "$(inputs.my_file.nameroot).md" -> "filename.md"

    :param to_resolve: The path to match
    :param inputs_to_reference: Inputs which are used to resolve input references like $(inputs.my_input_file.basename).

    :return: A string in which the input references are replaced with actual values.
    """

    splitted = split_input_references(to_resolve)

    result = []

    for part in splitted:
        if is_input_reference(part):
            result.append(str(resolve_input_reference(part, inputs_to_reference)))
        else:
            result.append(part)

    return ''.join(result)