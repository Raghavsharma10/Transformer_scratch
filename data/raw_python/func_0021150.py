def most_seen_works_card(kind=None, num=10):
    """
    Displays a card showing the Works that are associated with the most Events.
    """
    object_list = most_seen_works(kind=kind, num=num)

    object_list = chartify(object_list, 'num_views', cutoff=1)

    if kind:
        card_title = 'Most seen {}'.format(
                                    Work.get_kind_name_plural(kind).lower())
    else:
        card_title = 'Most seen works'

    return {
        'card_title': card_title,
        'score_attr': 'num_views',
        'object_list': object_list,
        'name_attr': 'title',
        'use_cite': True,
    }