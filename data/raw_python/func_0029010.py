def wire_deprecated(awsclient, function_name, s3_event_sources=None,
                    time_event_sources=None,
                    alias_name=ALIAS_NAME):
    """Deprecated! Please use wire!


    :param awsclient:
    :param function_name:
    :param s3_event_sources: dictionary
    :param time_event_sources:
    :param alias_name:
    :return: exit_code
    """
    if not lambda_exists(awsclient, function_name):
        log.error(colored.red('The function you try to wire up doesn\'t ' +
                          'exist... Bailing out...'))
        return 1
    client_lambda = awsclient.get_client('lambda')
    lambda_function = client_lambda.get_function(FunctionName=function_name)
    lambda_arn = client_lambda.get_alias(FunctionName=function_name,
                                         Name=alias_name)['AliasArn']
    log.info('wiring lambda_arn %s ...' % lambda_arn)

    if lambda_function is not None:
        s3_events_ensure_exists, s3_events_ensure_absent = filter_events_ensure(
            s3_event_sources)
        cloudwatch_events_ensure_exists, cloudwatch_events_ensure_absent = \
            filter_events_ensure(time_event_sources)

        for s3_event_source in s3_events_ensure_absent:
            _ensure_s3_event(awsclient, s3_event_source, function_name,
                             alias_name, lambda_arn, s3_event_source['ensure'])
        for s3_event_source in s3_events_ensure_exists:
            _ensure_s3_event(awsclient, s3_event_source, function_name,
                             alias_name, lambda_arn, s3_event_source['ensure'])

        for time_event in cloudwatch_events_ensure_absent:
            _ensure_cloudwatch_event(awsclient, time_event, function_name,
                                     alias_name, lambda_arn,
                                     time_event['ensure'])
        for time_event in cloudwatch_events_ensure_exists:
            _ensure_cloudwatch_event(awsclient, time_event, function_name,
                                     alias_name, lambda_arn,
                                     time_event['ensure'])
    return 0