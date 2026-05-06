def read_message_type(file):
    """Read the message type from a file."""
    message_byte = file.read(1)
    if message_byte == b'':
        return ConnectionClosed
    message_number = message_byte[0]
    return _message_types.get(message_number, UnknownMessage)