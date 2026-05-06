def _get_event_type(evt_source):
    """Get type of event e.g. 's3', 'events', 'kinesis',...

    :param evt_source:
    :return:
    """
    if 'schedule' in evt_source:
        return 'events'
    elif 'pattern' in evt_source:
        return 'events'
    elif 'log_group_name_prefix' in evt_source:
        return 'cloudwatch_logs'
    else:
        arn = evt_source['arn']
        _, _, svc, _ = arn.split(':', 3)
        return svc