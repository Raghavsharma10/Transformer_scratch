def summult(list1, list2):
    """
Multiplies elements in list1 and list2, element by element, and
returns the sum of all resulting multiplications.  Must provide equal
length lists.

Usage:   lsummult(list1,list2)
"""
    if len(list1) != len(list2):
        raise ValueError("Lists not equal length in summult.")
    s = 0
    for item1, item2 in zip(list1, list2):
        s = s + item1 * item2
    return s