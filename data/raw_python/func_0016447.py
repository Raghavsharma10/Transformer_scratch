def deep_update_dict(default, options):
    """
    Updates the values in a nested dict, while unspecified values will remain
    unchanged
    """
    for key in options.keys():
        default_setting = default.get(key)
        new_setting = options.get(key)
        if isinstance(default_setting, dict):
            deep_update_dict(default_setting, new_setting)
        else:
            default[key] = new_setting