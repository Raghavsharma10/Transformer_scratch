def adjust_status(info: dict) -> dict:
    """Apply status mapping to a raw API result."""
    modified_info = deepcopy(info)
    modified_info.update({
        'level':
            get_nearest_by_numeric_key(STATUS_MAP, int(info['level'])),
        'level2':
            STATUS_MAP[99] if info['level2'] is None else
            get_nearest_by_numeric_key(STATUS_MAP, int(info['level2']))
    })

    return modified_info