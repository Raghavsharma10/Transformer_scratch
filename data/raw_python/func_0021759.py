def schema(name, dct, *, strict=False):
    """Create a compound tag schema.

    This function is a short convenience function that makes it easy to
    subclass the base `CompoundSchema` class.

    The `name` argument is the name of the class and `dct` should be a
    dictionnary containing the actual schema. The schema should map keys
    to tag types or other compound schemas.

    If the `strict` keyword only argument is set to True, interacting
    with keys that are not defined in the schema will raise a
    `TypeError`.
    """
    return type(name, (CompoundSchema,), {'__slots__': (), 'schema': dct,
                                          'strict': strict})