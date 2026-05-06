def DosDateTimeToTimeTuple(dosDateTime):
    """Convert an MS-DOS format date time to a Python time tuple.
    """
    dos_date = dosDateTime >> 16
    dos_time = dosDateTime & 0xffff
    day = dos_date & 0x1f
    month = (dos_date >> 5) & 0xf
    year = 1980 + (dos_date >> 9)
    second = 2 * (dos_time & 0x1f)
    minute = (dos_time >> 5) & 0x3f
    hour = dos_time >> 11
    return time.localtime(
        time.mktime((year, month, day, hour, minute, second, 0, 1, -1)))