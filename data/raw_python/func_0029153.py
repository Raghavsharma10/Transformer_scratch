def deploy_stack(awsclient, context, conf, cloudformation, override_stack_policy=False):
    """Deploy the stack to AWS cloud. Does either create or update the stack.

    :param conf:
    :param override_stack_policy:
    :return: exit_code
    """
    stack_name = _get_stack_name(conf)
    parameters = _generate_parameters(conf)
    if stack_exists(awsclient, stack_name):
        exit_code = _update_stack(awsclient, context, conf, cloudformation,
                                  parameters, override_stack_policy)
    else:
        exit_code = _create_stack(awsclient, context, conf, cloudformation,
                                  parameters)
    # add 'stack_output' to the context so it becomes available
    # in 'command_finalized' hook
    context['stack_output'] = _get_stack_outputs(
        awsclient.get_client('cloudformation'), stack_name)
    _call_hook(awsclient, conf, stack_name, parameters, cloudformation,
               hook='post_hook',
               message='CloudFormation is done, now executing post hook...')
    return exit_code