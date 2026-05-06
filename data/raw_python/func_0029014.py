def check_and_format_logs_params(start, end, tail):
    """Helper to read the params for the logs command"""
    def _decode_duration_type(duration_type):
        durations = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
        return durations[duration_type]

    if not start:
        if tail:
            start_dt = maya.now().subtract(seconds=300).datetime(naive=True)
        else:
            start_dt = maya.now().subtract(days=1).datetime(naive=True)
    elif start and start[-1] in ['m', 'h', 'd', 'w']:
        value = int(start[:-1])
        start_dt = maya.now().subtract(
            **{_decode_duration_type(start[-1]): value}).datetime(naive=True)
    elif start:
        start_dt = maya.parse(start).datetime(naive=True)

    if end and end[-1] in ['m', 'h', 'd', 'w']:
        value = int(end[:-1])
        end_dt = maya.now().subtract(
            **{_decode_duration_type(end[-1]): value}).datetime(naive=True)
    elif end:
        end_dt = maya.parse(end).datetime(naive=True)
    else:
        end_dt = None
    return start_dt, end_dt