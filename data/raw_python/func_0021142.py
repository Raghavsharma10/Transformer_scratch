def annual_event_counts(kind='all'):
    """
    Returns a QuerySet of dicts, each one with these keys:

        * year - a date object representing the year
        * total - the number of events of `kind` that year

    kind - The Event `kind`, or 'all' for all kinds (default).
    """
    qs = Event.objects

    if kind != 'all':
        qs = qs.filter(kind=kind)

    qs = qs.annotate(year=TruncYear('date')) \
            .values('year') \
            .annotate(total=Count('id')) \
            .order_by('year')

    return qs