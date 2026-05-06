def get_tohu_items_name(cls):
    """
    Return a string which defines the name of the namedtuple class which will be used
    to produce items for the custom generator.

    By default this will be the first part of the class name (before '...Generator'),
    for example:

        FoobarGenerator -> Foobar
        QuuxGenerator   -> Quux

    However, it can be set explicitly by the user by defining `__tohu_items_name__`
    in the class definition, for example:

        class Quux(CustomGenerator):
            __tohu_items_name__ = 'MyQuuxItem'
    """
    assert issubclass(cls, TohuBaseGenerator)

    try:
        tohu_items_name = cls.__dict__['__tohu_items_name__']
        logger.debug(f"Using item class name '{tohu_items_name}' (derived from attribute '__tohu_items_name__')")
    except KeyError:
        m = re.match('^(.*)Generator$', cls.__name__)
        if m is not None:
            tohu_items_name = m.group(1)
            logger.debug(f"Using item class name '{tohu_items_name}' (derived from custom generator name)")
        else:
            msg = (
                "Cannot derive class name for items to be produced by custom generator. "
                "Please set '__tohu_items_name__' at the top of the custom generator's "
                "definition or change its name so that it ends in '...Generator'"
            )
            raise ValueError(msg)

    return tohu_items_name