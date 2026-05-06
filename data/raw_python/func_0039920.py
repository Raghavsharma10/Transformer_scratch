def uuid_to_date(uuid, century='20'):
    """Return a date created from the last 6 digits of a uuid.

    Arguments:
        uuid The unique identifier to parse.
        century The first 2 digits to assume in the year. Default is '20'.

    Examples:
        >>> uuid_to_date('e8820616-1462-49b6-9784-e99a32120201')
        datetime.date(2012, 2, 1)

        >>> uuid_to_date('e8820616-1462-49b6-9784-e99a32120201', '18')
        datetime.date(1812, 2, 1)

    """
    day = int(uuid[-2:])
    month = int(uuid[-4:-2])
    year = int('%s%s' % (century, uuid[-6:-4]))

    return datetime.date(year=year, month=month, day=day)