def set_debug(enabled: bool):
    """Enable or disable debug logs for the entire package.

    Parameters
    ----------
    enabled: bool
        Whether debug should be enabled or not.

    """
    global _DEBUG_ENABLED

    if not enabled:
        log('Disabling debug output...', logger_name=_LOGGER_NAME)
        _DEBUG_ENABLED = False
    else:
        _DEBUG_ENABLED = True
        log('Enabling debug output...', logger_name=_LOGGER_NAME)