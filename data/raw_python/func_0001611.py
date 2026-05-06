def as_dict(config):
    """
    Converts a ConfigParser object into a dictionary.

    The resulting dictionary has sections as keys which point to a dict of the
    sections options as key => value pairs.
    """
    settings = defaultdict(lambda: {})
    for section in config.sections():
        for key, val in config.items(section):
            settings[section][key] = val
    return settings