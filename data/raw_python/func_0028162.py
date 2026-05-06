def increment_name(name: str, start_marker: str = " (",
                   end_marker: str = ")") -> str:
    """
    Increment the name where the incremental part is given by parameters.

    Parameters
    ----------
    name : str, nbformat.notebooknode.NotebookNode
        Name
    start_marker : str
        The marker used before the incremental
    end_marker : str
        The marker after the incrementa

    Returns
    -------
    str
        Incremented name.

    >>> increment_name('abc')
    'abc (1)'
    >>> increment_name('abc(1)')
    'abc(1) (1)'
    >>> increment_name('abc (123)')
    'abc (124)'
    >>> increment_name('abc-1',start_marker='-',end_marker='')
    'abc-2'
    >>> increment_name('abc[2]',start_marker='[',end_marker=']')
    'abc[3]'
    >>> increment_name('abc1',start_marker='',end_marker='')
    Traceback (most recent call last):
        ...
    ValueError: start_marker can not be the empty string.
    """
    if start_marker == '':
        raise ValueError("start_marker can not be the empty string.")
    a = name
    start = len(a)-a[::-1].find(start_marker[::-1])

    if (a[len(a)-len(end_marker):len(a)] == end_marker
            and start < (len(a)-len(end_marker))
            and a[start-len(start_marker):start] == start_marker
            and a[start:len(a)-len(end_marker)].isdigit()):

        old_int = int(a[start:len(a)-len(end_marker)])
        new_int = old_int+1
        new_name = a[:start]+str(new_int)+end_marker
    else:
        new_name = a+start_marker+'1'+end_marker
    return new_name