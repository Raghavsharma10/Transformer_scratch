def vocab_windows_pebinary_type(instance):
    """Ensure file objects with the windows-pebinary-ext extension have a
    'pe-type' property that is from the windows-pebinary-type-ov vocabulary.
    """
    for key, obj in instance['objects'].items():
        if 'type' in obj and obj['type'] == 'file':
            try:
                pe_type = obj['extensions']['windows-pebinary-ext']['pe_type']
            except KeyError:
                continue
            if pe_type not in enums.WINDOWS_PEBINARY_TYPE_OV:
                yield JSONError("Object '%s' has a Windows PE Binary File "
                                "extension with a 'pe_type' of '%s', which is not a "
                                "value in the windows-pebinary-type-ov vocabulary."
                                % (key, pe_type), instance['id'],
                                'windows-pebinary-type')