def keep_everything_scorer(checked_ids):
    """Returns every query and every match in checked_ids, with best score."""
    result = checked_ids.keys()
    for i in checked_ids.values():
        result.extend(i.keys())
    return dict.fromkeys(result).keys()