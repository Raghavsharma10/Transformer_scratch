def most_visited_venues_card(num=10):
    """
    Displays a card showing the Venues that have the most Events.

    In spectator_core tags, rather than spectator_events so it can still be
    used on core pages, even if spectator_events isn't installed.
    """
    if spectator_apps.is_enabled('events'):

        object_list = most_visited_venues(num=num)

        object_list = chartify(object_list, 'num_visits', cutoff=1)

        return {
            'card_title': 'Most visited venues',
            'score_attr': 'num_visits',
            'object_list': object_list,
        }