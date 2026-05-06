def format_baseline_list(baseline_list):
    """Format the list of baseline information from the loaded files into a
    cohesive, informative string

    Parameters
    ------------
    baseline_list : (list)
        List of strings specifying the baseline information for each
        SuperMAG file

    Returns
    ---------
    base_string : (str)
        Single string containing the relevent data
        
    """

    uniq_base = dict()
    uniq_delta = dict()

    for bline in baseline_list:
        bsplit = bline.split()
        bdate = " ".join(bsplit[2:])
        if bsplit[0] not in uniq_base.keys():
            uniq_base[bsplit[0]] = ""
        if bsplit[1] not in uniq_delta.keys():
            uniq_delta[bsplit[1]] = ""

        uniq_base[bsplit[0]] += "{:s}, ".format(bdate)
        uniq_delta[bsplit[1]] += "{:s}, ".format(bdate)

    if len(uniq_base.items()) == 1:
        base_string = "Baseline {:s}".format(list(uniq_base.keys())[0])
    else:
        base_string = "Baseline "

        for i,kk in enumerate(uniq_base.keys()):
            if i == 1:
                base_string += "{:s}: {:s}".format(kk, uniq_base[kk][:-2])
            else:
                base_string += "         {:s}: {:s}".format(kk,
                                                            uniq_base[kk][:-2])
        else:
            base_string += "unknown"

    if len(uniq_delta.items()) == 1:
        base_string += "\nDelta    {:s}".format(list(uniq_delta.keys())[0])
    else:
        base_string += "\nDelta    "

        for i,kk in enumerate(uniq_delta.keys()):
            if i == 1:
                base_string += "{:s}: {:s}".format(kk, uniq_delta[kk][:-2])
            else:
                base_string += "         {:s}: {:s}".format(kk,
                                                            uniq_delta[kk][:-2])
        else:
            base_string += "unknown"

    return base_string