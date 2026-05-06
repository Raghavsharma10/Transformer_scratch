def camel_shuffle(sequence):
    """
    Inspired by https://stackoverflow.com/q/42549212/610569

        >>> list(range(12)) # Linear.
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        >>> camel_shuffle(list(range(12))) # M-shape.
        [0, 4, 8, 9, 5, 1, 2, 6, 10, 11, 7, 3]

        >>> camel_shuffle(list(reversed(range(12)))) #W-shape.
        [11, 7, 3, 2, 6, 10, 9, 5, 1, 0, 4, 8]
    """
    one_three, two_four = zigzag(sequence)
    one, three = zigzag(one_three)
    two, four = zigzag(two_four)
    return one + list(reversed(two)) + three + list(reversed(four))