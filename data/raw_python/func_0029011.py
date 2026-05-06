def unwire_deprecated(awsclient, function_name, s3_event_sources=None,
                      time_event_sources=None, alias_name=ALIAS_NAME):
    """Deprecated! Please use unwire!

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
    log.info('UN-wiring lambda_arn %s ' % lambda_arn)
    policies = None
    try:
        result = client_lambda.get_policy(FunctionName=function_name,
                                          Qualifier=alias_name)
        policies = json.loads(result['Policy'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            log.warn("Permission policies not found")
        else:
            raise e

    if lambda_function is not None:
        #### S3 Events
        # for every permission - delete it and corresponding rule (if exists)
        if policies:
            for statement in policies['Statement']:
                if statement['Principal']['Service'] == 's3.amazonaws.com':
                    source_bucket = get_bucket_from_s3_arn(
                        statement['Condition']['ArnLike']['AWS:SourceArn'])
                    log.info('\tRemoving S3 permission {} invoking {}'.format(
                        source_bucket, lambda_arn))
                    _remove_permission(awsclient, function_name,
                                       statement['Sid'], alias_name)
                    log.info('\tRemoving All S3 events {} invoking {}'.format(
                        source_bucket, lambda_arn))
                    _remove_events_from_s3_bucket(awsclient, source_bucket,
                                                  lambda_arn)

        # Case: s3 events without permissions active "safety measure"
        for s3_event_source in s3_event_sources:
            bucket_name = s3_event_source.get('bucket')
            _remove_events_from_s3_bucket(awsclient, bucket_name, lambda_arn)

        #### CloudWatch Events
        # for every permission - delete it and corresponding rule (if exists)
        if policies:
            for statement in policies['Statement']:
                if statement['Principal']['Service'] == 'events.amazonaws.com':
                    rule_name = get_rule_name_from_event_arn(
                        statement['Condition']['ArnLike']['AWS:SourceArn'])
                    log.info(
                        '\tRemoving Cloudwatch permission {} invoking {}'.format(
                            rule_name, lambda_arn))
                    _remove_permission(awsclient, function_name,
                                       statement['Sid'], alias_name)
                    log.info('\tRemoving Cloudwatch rule {} invoking {}'.format(
                        rule_name, lambda_arn))
                    _remove_cloudwatch_rule_event(awsclient, rule_name,
                                                  lambda_arn)
        # Case: rules without permissions active, "safety measure"
        for time_event in time_event_sources:
            rule_name = time_event.get('ruleName')
            _remove_cloudwatch_rule_event(awsclient, rule_name, lambda_arn)

    return 0