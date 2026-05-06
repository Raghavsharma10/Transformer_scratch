def sample_group(sid, groups):
    """
    Iterate through all categories in an OrderedDict and return category name if SampleID
    present in that category.

    :type sid: str
    :param sid: SampleID from dataset.

    :type groups: OrderedDict
    :param groups: Returned dict from phylotoast.util.gather_categories() function.

    :return type: str
    :return: Category name used to classify `sid`.
    """
    for name in groups:
        if sid in groups[name].sids:
            return name