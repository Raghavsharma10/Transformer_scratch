def merge_dictionaries(dicts, merge_lists=False):
    # type: (List[DictUpperBound], bool) -> DictUpperBound
    """Merges all dictionaries in dicts into a single dictionary and returns result

    Args:
        dicts (List[DictUpperBound]): Dictionaries to merge into the first one in the list
        merge_lists (bool): Whether to merge lists (True) or replace lists (False). Default is False.

    Returns:
        DictUpperBound: Merged dictionary

    """
    dict1 = dicts[0]
    for other_dict in dicts[1:]:
        merge_two_dictionaries(dict1, other_dict, merge_lists=merge_lists)
    return dict1