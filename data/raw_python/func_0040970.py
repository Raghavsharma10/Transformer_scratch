def equal(obj1, obj2):
    """Calculate equality between two (Comparable) objects."""
    Comparable.log(obj1, obj2, '==')
    equality = obj1.equality(obj2)
    Comparable.log(obj1, obj2, '==', result=equality)
    return equality