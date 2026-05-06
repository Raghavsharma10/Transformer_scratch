def prepare_logfile(filename: str) -> str:
    """Prepare an empty log file eventually and return its absolute path.

    When passing the "filename" `stdout`, |prepare_logfile| does not
    prepare any file and just returns `stdout`:

    >>> from hydpy.exe.commandtools import prepare_logfile
    >>> prepare_logfile('stdout')
    'stdout'

    When passing the "filename" `default`, |prepare_logfile| generates a
    filename containing the actual date and time, prepares an empty file
    on disk, and returns its path:

    >>> from hydpy import repr_, TestIO
    >>> from hydpy.core.testtools import mock_datetime_now
    >>> from datetime import datetime
    >>> with TestIO():
    ...     with mock_datetime_now(datetime(2000, 1, 1, 12, 30, 0)):
    ...         filepath = prepare_logfile('default')
    >>> import os
    >>> os.path.exists(filepath)
    True
    >>> repr_(filepath)    # doctest: +ELLIPSIS
    '...hydpy/tests/iotesting/hydpy_2000-01-01_12-30-00.log'

    For all other strings, |prepare_logfile| does not add any date or time
    information to the filename:

    >>> with TestIO():
    ...     with mock_datetime_now(datetime(2000, 1, 1, 12, 30, 0)):
    ...         filepath = prepare_logfile('my_log_file.txt')
    >>> os.path.exists(filepath)
    True
    >>> repr_(filepath)    # doctest: +ELLIPSIS
    '...hydpy/tests/iotesting/my_log_file.txt'
    """
    if filename == 'stdout':
        return filename
    if filename == 'default':
        filename = datetime.datetime.now().strftime(
            'hydpy_%Y-%m-%d_%H-%M-%S.log')
    with open(filename, 'w'):
        pass
    return os.path.abspath(filename)