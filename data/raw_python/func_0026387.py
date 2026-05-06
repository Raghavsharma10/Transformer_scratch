def overlapping(start1, end1, start2, end2):
    """
    >>> overlapping(0, 5, 6, 7)
    False
    >>> overlapping(1, 2, 0, 4)
    True
    >>> overlapping(5,6,0,5)
    False
    """
    return not ((start1 <= start2 and start1 <= end2 and end1 <= end2 and end1 <= start2) or
                (start1 >= start2 and start1 >= end2 and end1 >= end2 and end1 >= start2))