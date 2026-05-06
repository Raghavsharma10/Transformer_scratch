def zipLists(*lists):
    """
    Checks to see if all of the lists are the same length, and throws
    an AssertionError otherwise.  Returns the zipped lists.
    """
    length = len(lists[0])
    for i, list_ in enumerate(lists[1:]):
        if len(list_) != length:
            msg = "List at index {} has length {} != {}".format(
                i + 1, len(list_), length)
            raise AssertionError(msg)
    return zip(*lists)