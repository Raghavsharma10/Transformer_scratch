def find_field_generators(obj):
    """
    Return dictionary with the names and instances of
    all tohu.BaseGenerator occurring in the given
    object's class & instance namespaces.
    """

    cls_dict = obj.__class__.__dict__
    obj_dict = obj.__dict__
    #debug_print_dict(cls_dict, 'cls_dict')
    #debug_print_dict(obj_dict, 'obj_dict')

    field_gens = {}
    add_field_generators(field_gens, cls_dict)
    add_field_generators(field_gens, obj_dict)

    return field_gens