def get_obj_cacheable(obj, attr_name, calculate, recalculate=False):
    """
    Gets the result of a method call, using the given object and attribute name
    as a cache
    """
    if not recalculate and hasattr(obj, attr_name):
        return getattr(obj, attr_name)

    calculated = calculate()
    setattr(obj, attr_name, calculated)

    return calculated