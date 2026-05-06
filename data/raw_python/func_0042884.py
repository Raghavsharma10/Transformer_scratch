def dict_partial_cmp(target_dict, dict_list, ducktype):
    """
    Whether partial dict are in dict_list or not
    """
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
        if intersection == target_dict:
            return True
    # if no any arguments matched to called_args, return False
    return False