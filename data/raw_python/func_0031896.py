def search_greater(values, target):
    """
    Return the first index for which target is greater or equal to the first
    item of the tuple found in values
    """
    first = 0
    last = len(values)

    while first < last:
        middle = (first + last) // 2
        if values[middle][0] < target:
            first = middle + 1
        else:
            last = middle

    return first