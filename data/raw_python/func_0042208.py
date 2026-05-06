def _switch_charset_dict(characters, target=''):
    '''
    Switches the character set of the key/value pairs in a dictionary.
    '''
    offset_characters = {}
    offset = block_offset * offsets[target]['direction']
    for char in characters:
        offset_key = chr(ord(char) + offset)
        offset_value = chr(ord(characters[char]) + offset)
        offset_characters[offset_key] = offset_value

    return offset_characters