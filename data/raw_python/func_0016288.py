def get_setting(key, *default):
    """Return specific search setting from Django conf."""
    if default:
        return get_settings().get(key, default[0])
    else:
        return get_settings()[key]