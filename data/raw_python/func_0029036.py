def put_log_events(awsclient, log_group_name, log_stream_name, log_events,
                   sequence_token=None):
    """Put log events for the specified log group and stream.

    :param log_group_name: log group name
    :param log_stream_name: log stream name
    :param log_events: [{'timestamp': 123, 'message': 'string'}, ...]
    :param sequence_token: the sequence token
    :return: next_token
    """
    client_logs = awsclient.get_client('logs')
    request = {
        'logGroupName': log_group_name,
        'logStreamName': log_stream_name,
        'logEvents': log_events
    }
    if sequence_token:
        request['sequenceToken'] = sequence_token

    response = client_logs.put_log_events(**request)
    if 'rejectedLogEventsInfo' in response:
        log.warn(response['rejectedLogEventsInfo'])
    if 'nextSequenceToken' in response:
        return response['nextSequenceToken']