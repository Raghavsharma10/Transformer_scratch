def create_glir_message(commands, array_serialization=None):
    """Create a JSON-serializable message of GLIR commands. NumPy arrays
    are serialized according to the specified method.

    Arguments
    ---------

    commands : list
        List of GLIR commands.
    array_serialization : string or None
        Serialization method for NumPy arrays. Possible values are:
            'binary' (default) : use a binary string
            'base64' : base64 encoded string of the array

    """
    # Default serialization method for NumPy arrays.
    if array_serialization is None:
        array_serialization = 'binary'
    # Extract the buffers.
    commands_modified, buffers = _extract_buffers(commands)
    # Serialize the modified commands (with buffer pointers) and the buffers.
    commands_serialized = [_serialize_command(command_modified)
                           for command_modified in commands_modified]
    buffers_serialized = [_serialize_buffer(buffer, array_serialization)
                          for buffer in buffers]
    # Create the final message.
    msg = {
        'msg_type': 'glir_commands',
        'commands': commands_serialized,
        'buffers': buffers_serialized,
    }
    return msg