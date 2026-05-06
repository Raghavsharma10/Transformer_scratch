def convert_uptime_string_seconds(uptime):
    '''Convert uptime strings to seconds. The string can be formatted various ways.'''
    regex_list = [
        # n years, n weeks, n days, n hours, n minutes where each of the fields except minutes
        # is optional. Additionally, can be either singular or plural
        (r"((?P<years>\d+) year(s)?,\s+)?((?P<weeks>\d+) week(s)?,\s+)?"
         r"((?P<days>\d+) day(s)?,\s+)?((?P<hours>\d+) "
         r"hour(s)?,\s+)?((?P<minutes>\d+) minute(s)?)"),
        # n days, HH:MM:SS where each field is required (except for days)
        (r"((?P<days>\d+) day(s)?,\s+)?"
         r"((?P<hours>\d+)):((?P<minutes>\d+)):((?P<seconds>\d+))"),
        # 7w6d5h4m3s where each field is optional
        (r"((?P<weeks>\d+)w)?((?P<days>\d+)d)?((?P<hours>\d+)h)?"
         r"((?P<minutes>\d+)m)?((?P<seconds>\d+)s)?"),
    ]
    regex_list = [re.compile(x) for x in regex_list]

    uptime_dict = {}
    for regex in regex_list:
        match = regex.search(uptime)
        if match:
            uptime_dict = match.groupdict()
            break

    uptime_seconds = 0
    for unit, value in uptime_dict.items():
        if value is not None:
            if unit == 'years':
                uptime_seconds += int(value) * 31536000
            elif unit == 'weeks':
                uptime_seconds += int(value) * 604800
            elif unit == 'days':
                uptime_seconds += int(value) * 86400
            elif unit == 'hours':
                uptime_seconds += int(value) * 3600
            elif unit == 'minutes':
                uptime_seconds += int(value) * 60
            elif unit == 'seconds':
                uptime_seconds += int(value)
            else:
                raise Exception('Unrecognized unit "{}" in uptime:{}'.format(unit, uptime))

    if not uptime_dict:
        raise Exception('Unrecognized uptime string:{}'.format(uptime))

    return uptime_seconds