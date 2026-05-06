def compare_name_component(list1, list2, settings, use_ratio=False):
    """
    Compare a list of names from a name component based on settings
    """
    if not list1[0] or not list2[0]:
        not_required = not settings['required']
        return not_required * 100 if use_ratio else not_required

    if len(list1) != len(list2):
        return False

    compare_func = _ratio_compare if use_ratio else _normal_compare
    return compare_func(list1, list2, settings)