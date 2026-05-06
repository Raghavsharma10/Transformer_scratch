def endpoint(value: Any) -> Any:
    """
    Convert a endpoint string to the corresponding Endpoint instance type

    :param value: Endpoint string or subclass
    :return:
    """
    if issubclass(type(value), Endpoint):
        return value
    elif isinstance(value, str):
        for api, cls in MANAGED_API.items():
            if value.startswith(api + " "):
                return cls.from_inline(value)
        return UnknownEndpoint.from_inline(value)
    else:
        raise TypeError("Cannot convert {0} to endpoint".format(value))