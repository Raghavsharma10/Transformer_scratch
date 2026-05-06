def country_field(key='country'):
    """Provides a select box for country selection"""

    country_list = list(countries)
    title_map = []
    for item in country_list:
        title_map.append({'value': item.alpha_3, 'name': item.name})

    widget = {
        'key': key,
        'type': 'uiselect',
        'titleMap': title_map
    }

    return widget