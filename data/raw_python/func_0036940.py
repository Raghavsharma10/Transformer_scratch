def radpress_get_markup_descriptions():
    """
    Provides markup options. It used for adding descriptions in admin and
    zen mode.

    :return: list
    """
    result = []
    for markup in get_markup_choices():
        markup_name = markup[0]
        result.append({
            'name': markup_name,
            'title': markup[1],
            'description': trim(get_reader(markup=markup_name).description)
        })
    return result