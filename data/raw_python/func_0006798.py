def find_all( source, substring, start=None, end=None, overlap=False ):
    """Return every location a substring can be found in a source string.

    source
        The source string to search.

    start
        Start offset to read from (default: start)

    end
        End offset to stop reading at (default: end)

    overlap
        Whether to return overlapping matches (default: false)
    """
    return [x for x in find_all_iter( source, substring, start, end, overlap )]