def make_item_class_for_custom_generator_class(cls):
    """
    cls:
        The custom generator class for which to create an item-class
    """
    clsname = cls.__tohu_items_name__
    attr_names = cls.field_gens.keys()
    return make_item_class(clsname, attr_names)