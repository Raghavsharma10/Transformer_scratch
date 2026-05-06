def _switch_charset_list(characters, target=''):
    '''
    Switches the character set of a list. If a character does not have
    an equivalent in the target script (e.g. ヹ when converting to hiragana),
    the original character is kept.
    '''
    # Copy the list to avoid modifying the existing one.
    characters = characters[:]
    offset = block_offset * offsets[target]['direction']
    for n in range(len(characters)):
        chars = list(characters[n])

        for m in range(len(chars)):
            char = chars[m]
            char_offset = ord(char) + offset
            # Verify that the offset character is within the valid range.
            if in_range(char_offset, target):
                chars[m] = chr(char_offset)
            else:
                chars[m] = char

        characters[n] = ''.join(chars)

    return characters