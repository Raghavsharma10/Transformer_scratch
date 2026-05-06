def _remove_lead_trail_false(bool_list):
    """Remove leading and trailing false's from a list"""
    # The internet can be a wonderful place...
    for i in (0, -1):
        while bool_list and not bool_list[i]:
            bool_list.pop(i)
    return bool_list