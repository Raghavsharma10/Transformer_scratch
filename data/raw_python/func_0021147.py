def most_seen_creators_card(event_kind=None, num=10):
    """
    Displays a card showing the Creators that are associated with the most Events.
    """
    object_list = most_seen_creators(event_kind=event_kind, num=num)

    object_list = chartify(object_list, 'num_events', cutoff=1)

    return {
        'card_title': 'Most seen people/groups',
        'score_attr': 'num_events',
        'object_list': object_list,
    }