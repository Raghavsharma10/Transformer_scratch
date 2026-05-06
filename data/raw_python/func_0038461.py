def FirstSlotSlicer(primary_query, secondary_query, limit=30):  # noqa
    """
    Inject the first object from a queryset into the first position of a reading list.

    :param primary_queryset: djes.LazySearch object. Default queryset for reading list.
    :param secondary_queryset: djes.LazySearch object. first result leads the reading_list.
    :return list: mixed reading list.
    """
    reading_list = SearchSlicer(limit=limit)
    reading_list.register_queryset(primary_query)
    reading_list.register_queryset(secondary_query, validator=lambda x: bool(x == 0))
    return reading_list