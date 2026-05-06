def annual_event_counts_card(kind='all', current_year=None):
    """
    Displays years and the number of events per year.

    kind is an Event kind (like 'cinema', 'gig', etc.) or 'all' (default).
    current_year is an optional date object representing the year we're already
        showing information about.
    """
    if kind == 'all':
        card_title = 'Events per year'
    else:
        card_title = '{} per year'.format(Event.get_kind_name_plural(kind))

    return {
            'card_title': card_title,
            'kind': kind,
            'years': annual_event_counts(kind=kind),
            'current_year': current_year
            }