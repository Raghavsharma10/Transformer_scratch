def find_previous(element, l):
    """
    find previous element in a sorted list

    >>> find_previous(0, [0])
    0
    >>> find_previous(2, [1, 1, 3])
    1
    >>> find_previous(0, [1, 2])
    >>> find_previous(1.5, [1, 2])
    1
    >>> find_previous(3, [1, 2])
    2
    """
    length = len(l)
    for index, current in enumerate(l):
        # current is the last element
        if length - 1 == index:
            return current

        # current is the first element
        if index == 0:
            if element < current:
                return None

        if current <= element < l[index+1]:
            return current