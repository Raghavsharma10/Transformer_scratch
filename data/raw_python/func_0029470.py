def write_uniques(path, prefix, uniques):
    """
    Given a path, the method writes out one file for each group name in the
    uniques dictionary with the file name in the pattern

        PATH/prefix_group.txt

    with each file containing the unique OTUIDs found when comparing that group
    to all the other groups in uniques.

    :type path: str
    :param path: Output files will be saved in this PATH.

    :type prefix: str
    :param prefix: Prefix name added in front of output filename.

    :type uniques: dict
    :param uniques: Output from unique_otus() function.
    """
    for group in uniques:
        fp = osp.join(path, "{}_{}.txt".format(prefix, group))
        with open(fp, "w") as outf:
            outf.write("\n".join(uniques[group]))