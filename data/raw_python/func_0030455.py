def pretty_time(s, granularity=3):
    """Pretty print time in seconds. COnverts the input time in seconds into a string with
    interval names, such as days, hours and minutes

    From:
    http://stackoverflow.com/a/24542445/1144479

    """

    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )

    def display_time(seconds, granularity=granularity):
        result = []

        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append('{} {}'.format(int(value), name))

        return ', '.join(result[:granularity])

    return display_time(s, granularity)