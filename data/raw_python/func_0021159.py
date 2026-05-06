def annual_reading_counts_card(kind='all', current_year=None):
    """
    Displays years and the number of books/periodicals read per year.

    kind is one of 'book', 'periodical', 'all' (default).
    current_year is an optional date object representing the year we're already
        showing information about.
    """
    if kind == 'book':
        card_title = 'Books per year'
    elif kind == 'periodical':
        card_title = 'Periodicals per year'
    else:
        card_title = 'Reading per year'

    return {
            'card_title': card_title,
            'kind': kind,
            'years': utils.annual_reading_counts(kind),
            'current_year': current_year
            }