def sanitizeStructTime(struct):
    """
    Convert struct_time tuples with possibly invalid values to valid
    ones by substituting the closest valid value.
    """
    maxValues = (9999, 12, 31, 23, 59, 59)
    minValues = (1, 1, 1, 0, 0, 0)
    newstruct = []
    for value, maxValue, minValue in zip(struct[:6], maxValues, minValues):
        newstruct.append(max(minValue, min(value, maxValue)))
    return tuple(newstruct) + struct[6:]