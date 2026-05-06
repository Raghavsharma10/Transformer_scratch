def sort_key_for_numeric_suffixes(path, sep='.', suffix_index=-2):
    """
    Sort files taking into account potentially absent suffixes like
        somefile.dcd
        somefile.1000.dcd
        somefile.2000.dcd

    To be used with sorted(..., key=callable).
    """
    chunks = path.split(sep)
    # Remove suffix from path and convert to int
    if chunks[suffix_index].isdigit():
        return sep.join(chunks[:suffix_index] + chunks[suffix_index+1:]), int(chunks[suffix_index])
    return path, 0