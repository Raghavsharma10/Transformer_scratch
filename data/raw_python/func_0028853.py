def merge_deep(dct1, dct2, merger=None):
    """
        Deep merge by this spec below
    :param dct1:
    :param dct2:
    :param merger Optional merger
    :return:
    """
    my_merger = merger or Merger(
        # pass in a list of tuples,with the
        # strategies you are looking to apply
        # to each type.
        [
            (list, ["append"]),
            (dict, ["merge"])
        ],
        # next, choose the fallback strategies,
        # applied to all other types:
        ["override"],
        # finally, choose the strategies in
        # the case where the types conflict:
        ["override"]
    )
    return my_merger.merge(dct1, dct2)