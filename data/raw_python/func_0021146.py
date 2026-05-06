def day_events_card(date):
    """
    Displays Events that happened on the supplied date.
    `date` is a date object.
    """
    d = date.strftime(app_settings.DATE_FORMAT)
    card_title = 'Events on {}'.format(d)
    return {
            'card_title': card_title,
            'event_list': day_events(date=date),
            }