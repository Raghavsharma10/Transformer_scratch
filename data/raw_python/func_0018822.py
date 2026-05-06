def progressbar(iterable, length=23):
    """Print a simple progress bar while processing the given iterable.

    Function |progressbar| does print the progress bar when option
    `printprogress` is activted:

    >>> from hydpy import pub
    >>> pub.options.printprogress = True

    You can pass an iterable object.  Say you want to calculate the the sum
    of all integer values from 1 to 100 and print the progress of the
    calculation.  Using function |range| (which returns a list in Python 2
    and an iterator in Python3, but both are fine), one just has to
    interpose function |progressbar|:

    >>> from hydpy.core.printtools import progressbar
    >>> x_sum = 0
    >>> for x in progressbar(range(1, 101)):
    ...     x_sum += x
        |---------------------|
        ***********************
    >>> x_sum
    5050

    To prevent possible interim print commands from dismembering the status
    bar, they are delayed until the status bar is complete.  For intermediate
    print outs of each fiftieth calculation, the result looks as follows:

    >>> x_sum = 0
    >>> for x in progressbar(range(1, 101)):
    ...     x_sum += x
    ...     if not x % 50:
    ...         print(x, x_sum)
        |---------------------|
        ***********************
    50 1275
    100 5050


    The number of characters of the progress bar can be changed:

    >>> for i in progressbar(range(100), length=50):
    ...     continue
        |------------------------------------------------|
        **************************************************

    But its maximum number of characters is restricted by the length of the
    given iterable:

    >>> for i in progressbar(range(10), length=50):
    ...     continue
        |--------|
        **********

    The smallest possible progress bar has two characters:

    >>> for i in progressbar(range(2)):
    ...     continue
        ||
        **

    For iterables of length one or zero, no progress bar is plottet:


    >>> for i in progressbar(range(1)):
    ...     continue


    The same is True when the `printprogress` option is inactivated:

    >>> pub.options.printprogress = False
    >>> for i in progressbar(range(100)):
    ...     continue
    """
    if hydpy.pub.options.printprogress and (len(iterable) > 1):
        temp_name = os.path.join(tempfile.gettempdir(),
                                 'HydPy_progressbar_stdout')
        temp_stdout = open(temp_name, 'w')
        real_stdout = sys.stdout
        try:
            sys.stdout = temp_stdout
            nmbstars = min(len(iterable), length)
            nmbcounts = len(iterable)/nmbstars
            indentation = ' '*max(_printprogress_indentation, 0)
            with PrintStyle(color=36, font=1, file=real_stdout):
                print('    %s|%s|\n%s    ' % (indentation,
                                              '-'*(nmbstars-2),
                                              indentation),
                      end='',
                      file=real_stdout)
                counts = 1.
                for next_ in iterable:
                    counts += 1.
                    if counts >= nmbcounts:
                        print(end='*', file=real_stdout)
                        counts -= nmbcounts
                    yield next_
        finally:
            try:
                temp_stdout.close()
            except BaseException:
                pass
            sys.stdout = real_stdout
            print()
            with open(temp_name, 'r') as temp_stdout:
                sys.stdout.write(temp_stdout.read())
            sys.stdout.flush()
    else:
        for next_ in iterable:
            yield next_