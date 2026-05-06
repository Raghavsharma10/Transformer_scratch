def to(items, output=None):
    """Present a collection of items as TSV

    The items in the collection can themselves be any iterable collection.
    (Single field structures should be represented as one tuples.)

    With no output parameter, a generator of strings is returned. If an output
    parameter is passed, it should be a file-like object. Output is always
    newline separated.
    """
    strings = (format_collection(item) for item in items)
    if output is None:
        return strings
    else:
        for s in strings:
            output.write(s + '\n')