def set_item_class_name(cls_obj):
    """
    Return the first part of the class name of this custom generator.
    This will be used for the class name of the items produced by this
    generator.

    Examples:
        FoobarGenerator -> Foobar
        QuuxGenerator   -> Quux
    """
    if '__tohu__items__name__' in cls_obj.__dict__:
        logger.debug(f"Using item class name '{cls_obj.__tohu_items_name__}' (derived from attribute '__tohu_items_name__')")
    else:
        m = re.match('^(.*)Generator$', cls_obj.__name__)
        if m is not None:
            cls_obj.__tohu_items_name__ = m.group(1)
            logger.debug(f"Using item class name '{cls_obj.__tohu_items_name__}' (derived from custom generator name)")
        else:
            raise ValueError("Cannot derive class name for items to be produced by custom generator. "
                             "Please set '__tohu_items_name__' at the top of the custom generator's "
                             "definition or change its name so that it ends in '...Generator'")