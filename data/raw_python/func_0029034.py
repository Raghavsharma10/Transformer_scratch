def create_log_group(awsclient, log_group_name):
    """Creates a log group with the specified name.

    :param log_group_name: log group name
    :return:
    """
    client_logs = awsclient.get_client('logs')

    response = client_logs.create_log_group(
        logGroupName=log_group_name,
    )