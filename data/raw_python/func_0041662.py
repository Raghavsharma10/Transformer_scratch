def ensure_str(data: Union[str, bytes]) -> str:
    """
    Convert data in str if data are bytes

    :param data: Data
    :rtype str:
    """
    if isinstance(data, bytes):
        return str(data, 'utf-8')

    return data