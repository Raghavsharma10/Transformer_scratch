def tuple_partial_cmp(target_tuple, tuple_list, ducktype):
    """
    Whether partial target_tuple are in tuple_list or not
    """
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
        if not dst:
            return True
    # if no any arguments matched to called_tuple, return False
    return False