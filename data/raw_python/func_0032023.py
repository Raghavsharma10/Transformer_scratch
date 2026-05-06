def coerce_types(T1, T2):
    """Coerce types T1 and T2 to a common type.

    Coercion is performed according to this table, where "N/A" means
    that a TypeError exception is raised.

    +----------+-----------+-----------+-----------+----------+
    |          | int       | Fraction  | Decimal   | float    |
    +----------+-----------+-----------+-----------+----------+
    | int      | int       | Fraction  | Decimal   | float    |
    | Fraction | Fraction  | Fraction  | N/A       | float    |
    | Decimal  | Decimal   | N/A       | Decimal   | float    |
    | float    | float     | float     | float     | float    |
    +----------+-----------+-----------+-----------+----------+

    Subclasses trump their parent class; two subclasses of the same
    base class will be coerced to the second of the two.

    """
    # Get the common/fast cases out of the way first.
    if T1 is T2: return T1
    if T1 is int: return T2
    if T2 is int: return T1
    # Subclasses trump their parent class.
    if issubclass(T2, T1): return T2
    if issubclass(T1, T2): return T1
    # Floats trump everything else.
    if issubclass(T2, float): return T2
    if issubclass(T1, float): return T1
    # Subclasses of the same base class give priority to the second.
    if T1.__base__ is T2.__base__: return T2
    # Otherwise, just give up.
    raise TypeError('cannot coerce types %r and %r' % (T1, T2))