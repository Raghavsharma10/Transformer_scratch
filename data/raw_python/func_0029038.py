def check_log_stream_exists(awsclient, log_group_name, log_stream_name):
    """Check

    :param log_group_name: log group name
    :param log_stream_name: log stream name
    :return: True / False
    """
    lg = describe_log_group(awsclient, log_group_name)
    if lg and lg['logGroupName'] == log_group_name:
        stream = describe_log_stream(awsclient, log_group_name, log_stream_name)
        if stream and stream['logStreamName'] == log_stream_name:
            return True
    return False