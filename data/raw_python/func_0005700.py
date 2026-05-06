def make_item_class_for_custom_generator(obj):
    """
    obj:
        The custom generator instance for which to create an item class
    """
    clsname = obj.__tohu_items_name__
    attr_names = obj.field_gens.keys()
    return make_item_class(clsname, attr_names)