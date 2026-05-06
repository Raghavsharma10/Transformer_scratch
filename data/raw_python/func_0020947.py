def _parse_date_rfc822(dateString):
    '''Parse an RFC822, RFC1123, RFC2822, or asctime-style date'''
    data = dateString.split()
    if not data:
        return None
    if data[0][-1] in (',', '.') or data[0].lower() in rfc822._daynames:
        del data[0]
    if len(data) == 4:
        s = data[3]
        i = s.find('+')
        if i > 0:
            data[3:] = [s[:i], s[i+1:]]
        else:
            data.append('')
        dateString = " ".join(data)
    # Account for the Etc/GMT timezone by stripping 'Etc/'
    elif len(data) == 5 and data[4].lower().startswith('etc/'):
        data[4] = data[4][4:]
        dateString = " ".join(data)
    if len(data) < 5:
        dateString += ' 00:00:00 GMT'
    tm = rfc822.parsedate_tz(dateString)
    if tm:
        # Jython doesn't adjust for 2-digit years like CPython does,
        # so account for it by shifting the year so that it's in the
        # range 1970-2069 (1970 being the year of the Unix epoch).
        if tm[0] < 100:
            tm = (tm[0] + (1900, 2000)[tm[0] < 70],) + tm[1:]
        return time.gmtime(rfc822.mktime_tz(tm))