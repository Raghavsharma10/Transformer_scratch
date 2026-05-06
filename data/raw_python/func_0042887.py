def tuple_partial_cmp_always(target_tuple, tuple_list, ducktype):
    """
    Whether partial target_tuple are always in tuple_list or not
    """
    res = []
    for called_tuple in tuple_list:
        # ignore invalid test case
        if len(target_tuple) > len(called_tuple):
            continue
        # loop all argument from "current arguments"
        dst = len(target_tuple)
        for idx, part_target_tuple in enumerate(target_tuple):
            # test current argument one by one, if matched to previous record, counter-1
            dtype = ducktype(part_target_tuple)
            if hasattr(dtype, "mtest"):
                if dtype.mtest(called_tuple[idx]):
                    dst = dst - 1
            else:
                if dtype == called_tuple[idx]:
                    dst = dst - 1
        # if counter is zero => arguments is partial matched => return True
        ret = True if not dst else False
        res.append(ret)
    # if no any arguments matched to called_tuple, return False
    return True if res and False not in res else False