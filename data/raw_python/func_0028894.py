def delete_lambda_deprecated(awsclient, function_name, s3_event_sources=[],
                             time_event_sources=[], delete_logs=False):
    # FIXME: mutable default arguments!
    """Deprecated: please use delete_lambda!

    :param awsclient:
    :param function_name:
    :param s3_event_sources:
    :param time_event_sources:
    :param delete_logs:
    :return: exit_code
    """
    unwire_deprecated(awsclient, function_name, s3_event_sources=s3_event_sources,
                      time_event_sources=time_event_sources,
                      alias_name=ALIAS_NAME)
    client_lambda = awsclient.get_client('lambda')
    response = client_lambda.delete_function(FunctionName=function_name)
    if delete_logs:
        log_group_name = '/aws/lambda/%s' % function_name
        delete_log_group(awsclient, log_group_name)

    # TODO remove event source first and maybe also needed for permissions
    log.info(json2table(response))
    return 0