def most_read_creators_card(num=10):
    """
    Displays a card showing the Creators who have the most Readings
    associated with their Publications.

    In spectator_core tags, rather than spectator_reading so it can still be
    used on core pages, even if spectator_reading isn't installed.
    """
    if spectator_apps.is_enabled('reading'):

        object_list = most_read_creators(num=num)

        object_list = chartify(object_list, 'num_readings', cutoff=1)

        return {
            'card_title': 'Most read authors',
            'score_attr': 'num_readings',
            'object_list': object_list,
        }