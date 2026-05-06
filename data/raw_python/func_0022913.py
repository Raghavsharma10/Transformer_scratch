def _extract_buffers(commands):
    """Extract all data buffers from the list of GLIR commands, and replace
    them by buffer pointers {buffer: <buffer_index>}. Return the modified list
    # of GILR commands and the list of buffers as well."""
    # First, filter all DATA commands.
    data_commands = [command for command in commands if command[0] == 'DATA']
    # Extract the arrays.
    buffers = [data_command[3] for data_command in data_commands]
    # Modify the commands by replacing the array buffers with pointers.
    commands_modified = list(commands)
    buffer_index = 0
    for i, command in enumerate(commands_modified):
        if command[0] == 'DATA':
            commands_modified[i] = command[:3] + \
                ({'buffer_index': buffer_index},)
            buffer_index += 1
    return commands_modified, buffers