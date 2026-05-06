def extract_list_from_list_of_dict(list_of_dict, key):
    # type: (List[DictUpperBound], Any) -> List
    """Extract a list by looking up key in each member of a list of dictionaries

    Args:
        list_of_dict (List[DictUpperBound]): List of dictionaries
        key (Any): Key to find in each dictionary

    Returns:
        List: List containing values returned from each dictionary

    """
    result = list()
    for dictionary in list_of_dict:
        result.append(dictionary[key])
    return result