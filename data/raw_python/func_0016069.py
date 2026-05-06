def language_contents(instance):
    """Ensure keys in Language Content's 'contents' dictionary are valid
    language codes, and that the keys in the sub-dictionaries match the rules
    for object property names.
    """
    if instance['type'] != 'language-content' or 'contents' not in instance:
        return

    for key, value in instance['contents'].items():
        if key not in enums.LANG_CODES:
            yield JSONError("Invalid key '%s' in 'contents' property must be"
                            " an RFC 5646 code" % key, instance['id'])
        for subkey, subvalue in value.items():
            if not PROPERTY_FORMAT_RE.match(subkey):
                yield JSONError("'%s' in '%s' of the 'contents' property is "
                                "invalid and must match a valid property name"
                                % (subkey, key), instance['id'], 'observable-dictionary-keys')