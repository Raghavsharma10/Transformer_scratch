def rotate(l, steps=1):
    r"""Rotates a list `l` `steps` to the left. Accepts
    `steps` > `len(l)` or < 0.

    >>> rotate([1,2,3])
    [2, 3, 1]
    >>> rotate([1,2,3,4],-2)
    [3, 4, 1, 2]
    >>> rotate([1,2,3,4],-5)
    [4, 1, 2, 3]
    >>> rotate([1,2,3,4],1)
    [2, 3, 4, 1]
    >>> l = [1,2,3]; rotate(l) is not l
    True
    """
    if len(l):
        steps %= len(l)
        if steps:
            res = l[steps:]
            res.extend(l[:steps])
    return res