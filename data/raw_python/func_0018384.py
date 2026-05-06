def first_every_last(iterable, first, every, last):
    """Execute something before the first item of iter, something else for each
    item, and a third thing after the last.

    If there are no items in the iterable, don't execute anything.

    """
    did_first = False
    for item in iterable:
        if not did_first:
            did_first = True
            first(item)
        every(item)
    if did_first:
        last(item)