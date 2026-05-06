def create_log_stream(awsclient, log_group_name, log_stream_name):
    """Creates a log stream for the specified log group.

    :param log_group_name: log group name
    :param log_stream_name: log stream name
    :return:
    """
    client_logs = awsclient.get_client('logs')

    response = client_logs.create_log_stream(
        logGroupName=log_group_name,
        logStreamName=log_stream_name
    )