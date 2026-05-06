def is_config_container(v):
    """
    checks whether v is of type list,dict or Config
    """

    cls = type(v)

    return (
        issubclass(cls, list) or
        issubclass(cls, dict) or
        issubclass(cls, Config)
    )