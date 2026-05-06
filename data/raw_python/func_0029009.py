def unwire(awsclient, events, lambda_name, alias_name=ALIAS_NAME):
    """Unwire a list of event from an AWS Lambda function.

    'events' is a list of dictionaries, where the dict must contains the
    'schedule' of the event as string, and an optional 'name' and 'description'.

    :param awsclient:
    :param events: list of events
    :param lambda_name:
    :param alias_name:
    :return: exit_code
    """
    if not lambda_exists(awsclient, lambda_name):
        log.error(colored.red('The function you try to wire up doesn\'t ' +
                          'exist... Bailing out...'))
        return 1

    client_lambda = awsclient.get_client('lambda')
    lambda_function = client_lambda.get_function(FunctionName=lambda_name)
    lambda_arn = client_lambda.get_alias(FunctionName=lambda_name,
                                         Name=alias_name)['AliasArn']
    log.info('UN-wiring lambda_arn %s ' % lambda_arn)
    # TODO why load the policies here?
    '''
    policies = None
    try:
        result = client_lambda.get_policy(FunctionName=lambda_name,
                                          Qualifier=alias_name)
        policies = json.loads(result['Policy'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            log.warn("Permission policies not found")
        else:
            raise e
    '''

    if lambda_function is not None:
        #_unschedule_events(awsclient, events, lambda_arn)
        for event in events:
            evt_source = event['event_source']
            _remove_event_source(awsclient, evt_source, lambda_arn)
    return 0