def counts(data):
    """
    Generate a table of sorted (value, frequency) pairs.
    """
    if data is None:
        raise TypeError('None is not iterable')
    table = collections.Counter(data).most_common()
    if not table:
        return table
        # Extract the values with the highest frequency.
    maxfreq = table[0][1]
    for i in range(1, len(table)):
        if table[i][1] != maxfreq:
            table = table[:i]
            break
    return table