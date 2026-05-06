def character_set(instance):
    """Ensure certain properties of cyber observable objects come from the IANA
    Character Set list.
    """
    char_re = re.compile(r'^[a-zA-Z0-9_\(\)-]+$')
    for key, obj in instance['objects'].items():
        if ('type' in obj and obj['type'] == 'directory' and 'path_enc' in obj):
            if enums.char_sets():
                if obj['path_enc'] not in enums.char_sets():
                    yield JSONError("The 'path_enc' property of object '%s' "
                                    "('%s') must be an IANA registered "
                                    "character set."
                                    % (key, obj['path_enc']), instance['id'])
            else:
                info("Can't reach IANA website; using regex for character_set.")
                if not char_re.match(obj['path_enc']):
                    yield JSONError("The 'path_enc' property of object '%s' "
                                    "('%s') must be an IANA registered "
                                    "character set."
                                    % (key, obj['path_enc']), instance['id'])

        if ('type' in obj and obj['type'] == 'file' and 'name_enc' in obj):
            if enums.char_sets():
                if obj['name_enc'] not in enums.char_sets():
                    yield JSONError("The 'name_enc' property of object '%s' "
                                    "('%s') must be an IANA registered "
                                    "character set."
                                    % (key, obj['name_enc']), instance['id'])
            else:
                info("Can't reach IANA website; using regex for character_set.")
                if not char_re.match(obj['name_enc']):
                    yield JSONError("The 'name_enc' property of object '%s' "
                                    "('%s') must be an IANA registered "
                                    "character set."
                                    % (key, obj['name_enc']), instance['id'])