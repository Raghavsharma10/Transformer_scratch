def _parse_date_onblog(dateString):
    '''Parse a string according to the OnBlog 8-bit date format'''
    m = _korean_onblog_date_re.match(dateString)
    if not m:
        return
    w3dtfdate = '%(year)s-%(month)s-%(day)sT%(hour)s:%(minute)s:%(second)s%(zonediff)s' % \
                {'year': m.group(1), 'month': m.group(2), 'day': m.group(3),\
                 'hour': m.group(4), 'minute': m.group(5), 'second': m.group(6),\
                 'zonediff': '+09:00'}
    return _parse_date_w3dtf(w3dtfdate)