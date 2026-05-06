def compute_settings(args, rc_settings):
    """
    Merge arguments and rc_settings.
    """
    settings = {}
    for key, value in args.items():
        if key in ['reverse', 'opposite']:
            settings[key] = value ^ rc_settings.get(key, False)
        else:
            settings[key] = value or rc_settings.get(key)

    if not settings['size']:
        settings['size'] = DEFAULT_SIZE
    return settings