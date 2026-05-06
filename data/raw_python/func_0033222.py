def apply_command_list_template(command_list, in_path, out_path, args):
    '''
    Perform necessary substitutions on a command list to create a CLI-ready
    list to launch a conversion or download process via system binary.
    '''
    replacements = {
        '$IN': in_path,
        '$OUT': out_path,
    }

    # Add in positional arguments ($0, $1, etc)
    for i, arg in enumerate(args):
        replacements['$' + str(i)] = arg

    results = [replacements.get(arg, arg) for arg in command_list]

    # Returns list of truthy replaced arguments in command
    return [item for item in results if item]