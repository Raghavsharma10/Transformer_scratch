def set_item_class_name_on_custom_generator_class(cls):
    """
    Set the attribute `cls.__tohu_items_name__` to a string which defines the name
    of the namedtuple class which will be used to produce items for the custom
    generator.

    By default this will be the first part of the class name (before '...Generator'),
    for example:

        FoobarGenerator -> Foobar
        QuuxGenerator   -> Quux

    However, it can be set explicitly by the user by defining `__tohu_items_name__`
    in the class definition, for example:

        class Quux(CustomGenerator):
            __tohu_items_name__ = 'MyQuuxItem'
    """
    if '__tohu__items__name__' in cls.__dict__:
        logger.debug(
            f"Using item class name '{cls.__tohu_items_name__}' (derived from attribute '__tohu_items_name__')")
    else:
        m = re.match('^(.*)Generator$', cls.__name__)
        if m is not None:
            cls.__tohu_items_name__ = m.group(1)
            logger.debug(f"Using item class name '{cls.__tohu_items_name__}' (derived from custom generator name)")
        else:
            raise ValueError("Cannot derive class name for items to be produced by custom generator. "
                             "Please set '__tohu_items_name__' at the top of the custom generator's "
                             "definition or change its name so that it ends in '...Generator'")