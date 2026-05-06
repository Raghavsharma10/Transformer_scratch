def to_latin(string_to_transliterate, lang_code='sr'):
    ''' Transliterate serbian cyrillic string of characters to latin string of characters.
    :param string_to_transliterate: The cyrillic string to transliterate into latin characters.
    :param lang_code: Indicates the cyrillic language code we are translating from. Defaults to Serbian (sr).
    :return: A string of latin characters transliterated from the given cyrillic string.
    '''

    # First check if we support the cyrillic alphabet we want to transliterate to latin.
    if lang_code.lower() not in TRANSLIT_DICT:
        # If we don't support it, then just return the original string.
        return string_to_transliterate

    # If we do support it, check if the implementation is not missing before proceeding.
    elif not TRANSLIT_DICT[lang_code.lower()]['tolatin']:
        return string_to_transliterate

    # Everything checks out, proceed with transliteration.
    else:

        # Get the character per character transliteration dictionary
        transliteration_dict = TRANSLIT_DICT[lang_code.lower()]['tolatin']

        # Initialize the output latin string variable
        latinized_str = ''

        # Transliterate by traversing the input string character by character.
        string_to_transliterate = __decode_utf8(string_to_transliterate)


        for c in string_to_transliterate:

            # If character is in dictionary, it means it's a cyrillic so let's transliterate that character.
            if c in transliteration_dict:
                # Transliterate current character.
                latinized_str += transliteration_dict[c]

            # If character is not in character transliteration dictionary,
            # it is most likely a number or a special character so just keep it.
            else:
                latinized_str += c

        # Return the transliterated string.
        return __encode_utf8(latinized_str)