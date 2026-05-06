def print_progress(wrapped, _=None, args=None, kwargs=None):
    """Add print commands time to the given function informing about
     execution time.

    To show how the |print_progress| decorator works, we need to modify the
    functions used by |print_progress| to gain system time information
    available in module |time|.

    First, we mock the functions |time.strftime| and |time.perf_counter|:

    >>> import time
    >>> from unittest import mock
    >>> strftime = time.strftime
    >>> perf_counter = time.perf_counter
    >>> strftime_mock = mock.MagicMock()
    >>> time.strftime = strftime_mock
    >>> time.perf_counter = mock.MagicMock()

    The mock of |time.strftime| shall respond to two calls, as if the first
    call to a decorated function occurs at quarter past eight, and the second
    one two seconds later:

    >>> time.strftime.side_effect = '20:15:00', '20:15:02'

    The mock of |time.perf_counter| shall respond to four calls, as if the
    subsequent calls by decorated functions occur at second 1, 3, 4, and 7:

    >>> time.perf_counter.side_effect = 1, 3, 4, 7

    Now we decorate two test functions.  The first one does nothing; the
    second one only calls the first one:

    >>> from hydpy.core.printtools import print_progress
    >>> @print_progress
    ... def test1():
    ...     pass
    >>> @print_progress
    ... def test2():
    ...     test1()

    The first example shows that the output is appropriately indented,
    tat the returned times are at the right place, that the calculated
    execution the is correct, and that the mock of |time.strftime|
    received a valid format string:

    >>> from hydpy import pub
    >>> pub.options.printprogress = True
    >>> test2()
    method test2 started at 20:15:00
        method test1 started at 20:15:02
            seconds elapsed: 1
        seconds elapsed: 6
    >>> strftime_mock.call_args
    call('%H:%M:%S')

    The second example verifies that resetting the indentation works:

    >>> time.strftime.side_effect = '20:15:00', '20:15:02'
    >>> time.perf_counter.side_effect = 1, 3, 4, 7
    >>> test2()
    method test2 started at 20:15:00
        method test1 started at 20:15:02
            seconds elapsed: 1
        seconds elapsed: 6

    The last example shows that disabling the |Options.printprogress|
    option works as expected:

    >>> pub.options.printprogress = False
    >>> test2()

    >>> time.strftime = strftime
    >>> time.perf_counter = perf_counter
    """
    global _printprogress_indentation
    _printprogress_indentation += 4
    try:
        if hydpy.pub.options.printprogress:
            blanks = ' ' * _printprogress_indentation
            name = wrapped.__name__
            time_ = time.strftime('%H:%M:%S')
            with PrintStyle(color=34, font=1):
                print(f'{blanks}method {name} started at {time_}')
            seconds = time.perf_counter()
            sys.stdout.flush()
            wrapped(*args, **kwargs)
            blanks = ' ' * (_printprogress_indentation+4)
            seconds = time.perf_counter()-seconds
            with PrintStyle(color=34, font=1):
                print(f'{blanks}seconds elapsed: {seconds}')
            sys.stdout.flush()
        else:
            wrapped(*args, **kwargs)
    finally:
        _printprogress_indentation -= 4