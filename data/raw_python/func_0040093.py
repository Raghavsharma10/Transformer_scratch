def flatten(listish):
    """Flatten an arbitrarily-nested list of strings and lists.

    Works for any subclass of basestring and any type of iterable.
    """
    for elem in listish:
        if (isinstance(elem, collections.Iterable)
                and not isinstance(elem, basestring)):
            for subelem in flatten(elem):
                yield subelem
        else:
            yield elem