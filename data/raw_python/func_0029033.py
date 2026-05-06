def describe_log_stream(awsclient, log_group_name, log_stream_name):
    """Get info on the specified log stream

    :param log_group_name: log group name
    :param log_stream_name: log stream
    :return:
    """
    client_logs = awsclient.get_client('logs')

    response = client_logs.describe_log_streams(
        logGroupName=log_group_name,
        logStreamNamePrefix=log_stream_name,
        limit=1
    )
    if response['logStreams']:
        return response['logStreams'][0]
    else:
        return