def create_conf_loader(*args, **kwargs):  # pragma: no cover
    """Create a default configuration loader.

    .. deprecated:: 1.0.0b1
       Use :func:`create_config_loader` instead. This function will be removed
       in version 1.0.1.
    """
    import warnings
    warnings.warn(
        '"create_conf_loader" has been renamed to "create_config_loader".',
        DeprecationWarning
    )
    return create_config_loader(*args, **kwargs)