def process_char(buffer: str, char: str, mappings=_char_mappings):
    """This is a convinience method for use with 
    EventListener.wait_for_unicode_char(). In most cases it simply appends 
    char to buffer. Some replacements are done because presing return will
    produce '\\r' but for most cases '\\n' would be desireable.
    Also backspace cant just be added to a string either, therefore, if char is
    "\\u0008" the last character from buffer will be cut off. The replacement
    from '\\r' to '\\n' is done using the mappings argument, the default value
    for it also contains a mapping from '\t' to 4 spaces.

    :param buffer: the string to be updated
    :type buffer: str

    :param char: the unicode character to be processed
    :type char: str

    :param mappings: a dict containing mappings
    :type mappings: dict

    :returns: a new string"""
    if char in mappings:
        return buffer + mappings[char]
    elif char == "\u0008":
        return buffer[:-1] if len(buffer) > 0 else buffer
    else:
        return buffer + char