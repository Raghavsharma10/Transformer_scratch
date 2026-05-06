def area_field(key='area'):
    """Provides a select box for country selection"""

    area_list = list(subdivisions)
    title_map = []
    for item in area_list:
        title_map.append({'value': item.code, 'name': item.name})

    widget = {
        'key': key,
        'type': 'uiselect',
        'titleMap': title_map
    }

    return widget