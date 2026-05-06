def normalize_fieldsets(fieldsets):
    """
    Make sure the keys in fieldset dictionaries are strings. Returns the
    normalized data.
    """
    result = []
    for name, options in fieldsets:
        result.append((name, normalize_dictionary(options)))
    return result