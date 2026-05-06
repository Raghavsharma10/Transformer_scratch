def delete_stack(awsclient, conf, feedback=True):
    """Delete the stack from AWS cloud.

    :param awsclient:
    :param conf:
    :param feedback: print out stack events (defaults to True)
    """
    client_cf = awsclient.get_client('cloudformation')
    stack_name = _get_stack_name(conf)
    last_event = _get_stack_events_last_timestamp(awsclient, stack_name)

    request = {}
    dict_selective_merge(request, conf['stack'], ['StackName', 'RoleARN'])

    response = client_cf.delete_stack(**request)

    if feedback:
        return _poll_stack_events(awsclient, stack_name, last_event)