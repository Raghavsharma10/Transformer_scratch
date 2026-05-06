def event_list_tabs(counts, current_kind, page_number=1):
    """
    Displays the tabs to different event_list pages.

    `counts` is a dict of number of events for each kind, like:
        {'all': 30, 'gig': 12, 'movie': 18,}

    `current_kind` is the event kind that's active, if any. e.g. 'gig',
        'movie', etc.

    `page_number` is the current page of this kind of events we're on.
    """
    return {
            'counts': counts,
            'current_kind': current_kind,
            'page_number': page_number,
            # A list of all the kinds we might show tabs for, like
            # ['gig', 'movie', 'play', ...]
            'event_kinds': Event.get_kinds(),
            # A dict of data about each kind, keyed by kind ('gig') including
            # data about 'name', 'name_plural' and 'slug':
            'event_kinds_data': Event.get_kinds_data(),
        }