def load_ipa_data():
    """
    Load the IPA data from the built-in IPA database, creating the following globals:

    1. ``IPA_CHARS``: list of all IPAChar objects
    2. ``UNICODE_TO_IPA``: dict mapping a Unicode string (often, a single char) to an IPAChar
    3. ``UNICODE_TO_IPA_MAX_KEY_LENGTH``: length of a longest key in ``UNICODE_TO_IPA``
    4. ``IPA_TO_UNICODE``: map an IPAChar canonical representation to the corresponding Unicode string (or char)
    """
    ipa_signs = []
    unicode_to_ipa = {}
    ipa_to_unicode = {}
    max_key_length = 0
    for line in load_data_file(
        file_path=u"ipa.dat",
        file_path_is_relative=True,
        line_format=u"sU"
    ):
        # unpack data
        i_desc, i_unicode_keys = line
        name = re.sub(r" [ ]*", " ", i_desc)

        # create a suitable IPACharacter obj
        if u"consonant" in i_desc:
            obj = IPAConsonant(name=name, descriptors=i_desc)
        elif u"vowel" in i_desc:
            obj = IPAVowel(name=name, descriptors=i_desc)
        elif u"diacritic" in i_desc:
            obj = IPADiacritic(name=name, descriptors=i_desc)
        elif u"suprasegmental" in i_desc:
            obj = IPASuprasegmental(name=name, descriptors=i_desc)
        elif u"tone" in i_desc:
            obj = IPATone(name=name, descriptors=i_desc)
        else:
            raise ValueError("The IPA data file contains a bad line, defining an unknown type: '%s'" % (line))
        ipa_signs.append(obj)

        # map Unicode codepoint to object, if the former is available
        if len(i_unicode_keys) > 0:
            # canonical Unicode string
            first_key = i_unicode_keys[0]
            ipa_to_unicode[obj.canonical_representation] = first_key
            obj.unicode_repr = first_key
            max_key_length = max(max_key_length, len(first_key))
            # add all Unicode strings 
            for key in i_unicode_keys:
                if key in unicode_to_ipa:
                    raise ValueError("The IPA data file contains a bad line, redefining codepoint '%s': '%s'" % (key, line))
                unicode_to_ipa[key] = obj
    return ipa_signs, unicode_to_ipa, max_key_length, ipa_to_unicode