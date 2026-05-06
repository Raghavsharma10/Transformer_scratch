def get_internal_modules(key='exa'):
    """
    Get a list of modules belonging to the given package.

    Args:
        key (str): Package or library name (e.g. "exa")
    """
    key += '.'
    return [v for k, v in sys.modules.items() if k.startswith(key)]