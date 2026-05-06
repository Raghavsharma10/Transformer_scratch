def date_range(date, func='date'):
    '''
    Return back start and end dates given date string

    :param date: metrique date (range) to apply to pql query

    The tilde '~' symbol is used as a date range separated.

    A tilde by itself will mean 'all dates ranges possible'
    and will therefore search all objects irrelevant of it's
    _end date timestamp.

    A date on the left with a tilde but no date on the right
    will generate a query where the date range starts
    at the date provide and ends 'today'.
    ie, from date -> now.

    A date on the right with a tilde but no date on the left
    will generate a query where the date range starts from
    the first date available in the past (oldest) and ends
    on the date provided.
    ie, from beginning of known time -> date.

    A date on both the left and right will be a simple date
    range query where the date range starts from the date
    on the left and ends on the date on the right.
    ie, from date to date.
    '''
    if isinstance(date, basestring):
        date = date.strip()
    if not date:
        return '_end == None'
    if date == '~':
        return ''

    # don't include objects which have start EXACTLY on the
    # date in question, since we're looking for objects
    # which were true BEFORE the given date, not before or on.
    before = lambda d: '_start < %s("%s")' % (func, ts2dt(d) if d else None)
    after = lambda d: '(_end >= %s("%s") or _end == None)' % \
        (func, ts2dt(d) if d else None)
    split = date.split('~')
    # replace all occurances of 'T' with ' '
    # this is used for when datetime is passed in
    # like YYYY-MM-DDTHH:MM:SS instead of
    #      YYYY-MM-DD HH:MM:SS as expected
    # and drop all occurances of 'timezone' like substring
    # FIXME: need to adjust (to UTC) for the timezone info we're dropping!
    split = [re.sub('\+\d\d:\d\d', '', d.replace('T', ' ')) for d in split]
    if len(split) == 1:  # 'dt'
        return '%s and %s' % (before(split[0]), after(split[0]))
    elif split[0] in ['', None]:  # '~dt'
        return before(split[1])
    elif split[1] in ['', None]:  # 'dt~'
        return after(split[0])
    else:  # 'dt~dt'
        return '%s and %s' % (before(split[1]), after(split[0]))