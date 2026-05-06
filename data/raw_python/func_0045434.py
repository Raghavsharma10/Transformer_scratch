def _get_file_creation_time(file_path):
    """Returns the creation time of the file at the specified file path in Microsoft FILETIME
       structure format (https://msdn.microsoft.com/en-us/library/windows/desktop/ms724284.aspx),
       formatted as a 8-byte unsigned integer bytearray.
    """

    ctime = getctime(file_path)

    if ctime < -11644473600 or ctime >= 253402300800:
        raise FileTimeOutOfRangeException(ctime)

    creation_time_datetime = datetime.utcfromtimestamp(ctime)

    creation_time_epoch_offset = creation_time_datetime - datetime(1601, 1, 1)

    creation_time_secs_from_epoch = _convert_timedelta_to_seconds(creation_time_epoch_offset)

    creation_time_filetime = int(creation_time_secs_from_epoch * (10 ** 7))

    file_creation_time = bytearray(8)
    pack_into(b"Q", file_creation_time, 0, creation_time_filetime)

    return file_creation_time