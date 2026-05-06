def lengths_offsets(value):
    """Split the given comma separated value to multiple integer values. """
    values = []
    for item in value.split(','):
        item = int(item)
        values.append(item)
    return values