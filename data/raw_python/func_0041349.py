def block_uid(value: Union[str, BlockUID, None]) -> BlockUID:
    """
    Convert value to BlockUID instance

    :param value: Value to convert
    :return:
    """
    if isinstance(value, BlockUID):
        return value
    elif isinstance(value, str):
        return BlockUID.from_str(value)
    elif value is None:
        return BlockUID.empty()
    else:
        raise TypeError("Cannot convert {0} to BlockUID".format(type(value)))