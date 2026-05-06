def get_keys(data_list, leading_columns=LEADING_COLUMNS):
    """Gets all possible keys from a list of dicts, sorting by leading_columns first

    Args:
        data_list: list of dicts to pull keys from
        leading_columns: list of keys to put first in the result

    Returns:
        list of keys to be included as columns in excel worksheet
    """
    all_keys = set().union(*(list(d.keys()) for d in data_list))

    leading_keys = []

    for key in leading_columns:
        if key not in all_keys:
            continue
        leading_keys.append(key)
        all_keys.remove(key)

    return leading_keys + sorted(all_keys)