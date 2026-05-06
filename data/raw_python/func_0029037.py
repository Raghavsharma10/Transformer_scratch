def get_log_events(awsclient, log_group_name, log_stream_name, start_ts=None):
    """Get log events for the specified log group and stream.
    this is used in tenkai output instance diagnostics

    :param log_group_name: log group name
    :param log_stream_name: log stream name
    :param start_ts: timestamp
    :return:
    """
    client_logs = awsclient.get_client('logs')

    request = {
        'logGroupName': log_group_name,
        'logStreamName': log_stream_name
    }
    if start_ts:
        request['startTime'] = start_ts

    # TODO exhaust the events!
    # TODO use all_pages !
    response = client_logs.get_log_events(**request)

    if 'events' in response and response['events']:
        return [{'timestamp': e['timestamp'], 'message': e['message']}
                for e in response['events']]