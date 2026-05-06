def delete_lambda(awsclient, function_name, events=None, delete_logs=False):
    """Delete a lambda function.

    :param awsclient:
    :param function_name:
    :param events: list of events
    :param delete_logs:
    :return: exit_code
    """
    if events is not None:
        unwire(awsclient, events, function_name, alias_name=ALIAS_NAME)
    client_lambda = awsclient.get_client('lambda')
    response = client_lambda.delete_function(FunctionName=function_name)
    if delete_logs:
        log_group_name = '/aws/lambda/%s' % function_name
        delete_log_group(awsclient, log_group_name)

    # TODO remove event source first and maybe also needed for permissions
    log.info(json2table(response))
    return 0