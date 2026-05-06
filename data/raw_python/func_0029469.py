def shared_otuids(groups):
    """
    Get shared OTUIDs between all unique combinations of groups.

    :type groups: Dict
    :param groups: {Category name: OTUIDs in category}

    :return type: dict
    :return: Dict keyed on group combination and their shared OTUIDs as values.
    """
    for g in sorted(groups):
        print("Number of OTUs in {0}: {1}".format(g, len(groups[g].results["otuids"])))
    number_of_categories = len(groups)
    shared = defaultdict()
    for i in range(2, number_of_categories+1):
        for j in combinations(sorted(groups), i):
            combo_name = " & ".join(list(j))
            for grp in j:
                # initialize combo values
                shared[combo_name] = groups[j[0]].results["otuids"].copy()
                """iterate through all groups and keep updating combo OTUIDs with set
                intersection_update"""
                for grp in j[1:]:
                    shared[combo_name].intersection_update(groups[grp].results["otuids"])
    return shared