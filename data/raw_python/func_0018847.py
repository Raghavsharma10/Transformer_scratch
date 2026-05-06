def to_repr(self: Variable, values, brackets1d: Optional[bool] = False) \
        -> str:
    """Return a valid string representation for the given |Variable|
    object.

    Function |to_repr| it thought for internal purposes only, more
    specifically for defining string representations of subclasses
    of class |Variable| like the following:

    >>> from hydpy.core.variabletools import to_repr, Variable
    >>> class Var(Variable):
    ...     NDIM = 0
    ...     TYPE = int
    ...     __hydpy__connect_variable2subgroup__ = None
    ...     initinfo = 1.0, False
    >>> var = Var(None)
    >>> var.value = 2
    >>> var
    var(2)

    The following examples demonstrate all covered cases.  Note that
    option `brackets1d` allows choosing between a "vararg" and an
    "iterable" string representation for 1-dimensional variables
    (the first one being the default):

    >>> print(to_repr(var, 2))
    var(2)

    >>> Var.NDIM = 1
    >>> var = Var(None)
    >>> var.shape = 3
    >>> print(to_repr(var, range(3)))
    var(0, 1, 2)
    >>> print(to_repr(var, range(3), True))
    var([0, 1, 2])
    >>> print(to_repr(var, range(30)))
    var(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
    >>> print(to_repr(var, range(30), True))
    var([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])

    >>> Var.NDIM = 2
    >>> var = Var(None)
    >>> var.shape = (2, 3)
    >>> print(to_repr(var, [range(3), range(3, 6)]))
    var([[0, 1, 2],
         [3, 4, 5]])
    >>> print(to_repr(var, [range(30), range(30, 60)]))
    var([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
         [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]])
    """
    prefix = f'{self.name}('
    if isinstance(values, str):
        string = f'{self.name}({values})'
    elif self.NDIM == 0:
        string = f'{self.name}({objecttools.repr_(values)})'
    elif self.NDIM == 1:
        if brackets1d:
            string = objecttools.assignrepr_list(values, prefix, 72) + ')'
        else:
            string = objecttools.assignrepr_values(
                values, prefix, 72) + ')'
    else:
        string = objecttools.assignrepr_list2(values, prefix, 72) + ')'
    return '\n'.join(self.commentrepr + [string])