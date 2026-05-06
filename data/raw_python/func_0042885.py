def dict_partial_cmp_always(target_dict, dict_list, ducktype):
    """
    Whether partial dict are always in dict_list or not
    """
    res = []
    for called_dict in dict_list:
        # ignore invalid test case
        if len(target_dict) > len(called_dict):
            continue
        # get the intersection of two dicts
        intersection = {}
        for item in target_dict:
            dtype = ducktype(target_dict[item])
            if hasattr(dtype, "mtest"):
                if item in called_dict and dtype.mtest(called_dict[item]):
                    intersection[item] = target_dict[item]
            else:
                if item in called_dict and dtype == called_dict[item]:
                    intersection[item] = target_dict[item]
        ret = True if intersection == target_dict else False
        res.append(ret)
    # if no any arguments matched to called_args, return False
    return True if res and False not in res else False