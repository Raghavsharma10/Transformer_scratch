def tuple_in_list_always(main, sub):
    """
    >>> main = [('a', 'b', 'c'), ('c', 'd')]
    >>> tuple_in_list_always(main, ('a' ,'b', 'c'))
    False
    >>> tuple_in_list_always(main, ('c', 'd'))
    False
    >>> tuple_in_list_always(main, ('a', 'c'))
    False
    >>> tuple_in_list_always(main, ('a'))
    False
    >>> tuple_in_list_always(main, ((),))
    False
    >>> main = [('a', 'b', 'c'), ('a', 'b', 'c')]
    >>> tuple_in_list_always(main, ('a' ,'b', 'c'))
    True
    >>> main = [('a', 'b', 'c')]
    >>> tuple_in_list_always(main, ('a' ,'b', 'c'))
    True
    """
    return True if sub in set(main) and len(set(main)) == 1 else False