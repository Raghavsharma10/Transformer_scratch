def day_publications(date):
    """
    Returns a QuerySet of Publications that were being read on `date`.
    `date` is a date tobject.
    """
    readings = Reading.objects \
                        .filter(start_date__lte=date) \
                        .filter(
                            Q(end_date__gte=date)
                            |
                            Q(end_date__isnull=True)
                        )
    if readings:
        return Publication.objects.filter(reading__in=readings) \
                        .select_related('series') \
                        .prefetch_related('roles__creator') \
                        .distinct()
    else:
        return Publication.objects.none()