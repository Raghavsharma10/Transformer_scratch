def clean_up_datetime(obj_map):
    """convert datetime objects to dictionaries for storage"""
    clean_map = {}
    for key, value in obj_map.items():
        if isinstance(value, datetime.datetime):
            clean_map[key] = {
                'year': value.year,
                'month': value.month,
                'day': value.day,
                'hour': value.hour,
                'minute': value.minute,
                'second': value.second,
                'microsecond': value.microsecond,
                'tzinfo': value.tzinfo
            }
        elif isinstance(value, dict):
            clean_map[key] = clean_up_datetime(value)
        elif isinstance(value, list):
            if key not in clean_map:
                clean_map[key] = []
            if len(value) > 0:
                for index, list_value in enumerate(value):
                    if isinstance(list_value, dict):
                        clean_map[key].append(clean_up_datetime(list_value))
                    else:
                        clean_map[key].append(list_value)
            else:
                clean_map[key] = value
        else:
            clean_map[key] = value
    return clean_map