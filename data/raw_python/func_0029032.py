def describe_log_group(awsclient, log_group_name):
    """Get info on the specified log group

    :param log_group_name: log group name
    :return:
    """
    client_logs = awsclient.get_client('logs')

    request = {
        'logGroupNamePrefix': log_group_name,
        'limit': 1
    }
    response = client_logs.describe_log_groups(**request)
    if response['logGroups']:
        return response['logGroups'][0]
    else:
        return