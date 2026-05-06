def delete_log_group(awsclient, log_group_name):
    """Delete the specified log group

    :param log_group_name: log group name
    :return:
    """
    client_logs = awsclient.get_client('logs')

    response = client_logs.delete_log_group(
        logGroupName=log_group_name
    )