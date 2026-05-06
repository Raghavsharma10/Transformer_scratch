def unique_otuids(groups):
    """
    Get unique OTUIDs of each category.

    :type groups: Dict
    :param groups: {Category name: OTUIDs in category}

    :return type: dict
    :return: Dict keyed on category name and unique OTUIDs as values.
    """
    uniques = {key: set() for key in groups}
    for i, group in enumerate(groups):
        to_combine = groups.values()[:i]+groups.values()[i+1:]
        combined = combine_sets(*to_combine)
        uniques[group] = groups[group].difference(combined)
    return uniques