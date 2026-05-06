def info(awsclient, config, format=None):
    """
    collect info and output to console

    :param awsclient:
    :param config:
    :param json: True / False to use json format as output
    :return:
    """
    if format is None:
        format = 'tabular'
    stack_name = _get_stack_name(config)
    client_cfn = awsclient.get_client('cloudformation')

    resources = all_pages(
        client_cfn.list_stack_resources,
        {'StackName': stack_name},
        lambda x: [(r['ResourceType'], r['LogicalResourceId'], r['ResourceStatus'])
            for r in x['StackResourceSummaries']]
    )

    infos = {
        'stack_output': _get_stack_outputs(client_cfn, stack_name),
        'stack_state': _get_stack_state(client_cfn, stack_name),
        'resources': resources
    }
    if format == 'json':
        print(json.dumps(infos))
    elif format == 'tabular':
        print('stack output:')
        print(tabulate(infos['stack_output'], tablefmt='fancy_grid'))
        print('\nstack_state: %s' % infos['stack_state'])
        print('\nresources:')
        print(tabulate(infos['resources'], tablefmt='fancy_grid'))