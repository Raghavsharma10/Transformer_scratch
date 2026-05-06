def print_textandtime(text: str) -> None:
    """Print the given string and the current date and time with high
    precision for logging purposes.

    >>> from hydpy.exe.commandtools import print_textandtime
    >>> from hydpy.core.testtools import mock_datetime_now
    >>> from datetime import datetime
    >>> with mock_datetime_now(datetime(2000, 1, 1, 12, 30, 0, 123456)):
    ...     print_textandtime('something happens')
    something happens (2000-01-01 12:30:00.123456).
    """
    timestring = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f'{text} ({timestring}).')