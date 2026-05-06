def put_retention_policy(awsclient, log_group_name, retention_in_days):
    """Sets the retention of the specified log group
    if the log group does not yet exist than it will be created first.

    :param log_group_name: log group name
    :param retention_in_days: log group name
    :return:
    """
    try:
        # Note: for AWS Lambda the log_group is created once the first
        # log event occurs. So if the log_group does not exist we create it
        create_log_group(awsclient, log_group_name)
    except GracefulExit:
        raise
    except Exception:
        # TODO check that it is really a ResourceAlreadyExistsException
        pass

    client_logs = awsclient.get_client('logs')
    response = client_logs.put_retention_policy(
        logGroupName=log_group_name,
        retentionInDays=retention_in_days
    )