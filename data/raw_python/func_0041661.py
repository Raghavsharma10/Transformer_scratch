def ensure_bytes(data: Union[str, bytes]) -> bytes:
    """
    Convert data in bytes if data is a string

    :param data: Data
    :rtype bytes:
    """
    if isinstance(data, str):
        return bytes(data, 'utf-8')

    return data