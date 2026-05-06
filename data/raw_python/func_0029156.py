def describe_change_set(awsclient, change_set_name, stack_name):
    """Print out the change_set to console.
    This needs to run create_change_set first.

    :param awsclient:
    :param change_set_name:
    :param stack_name:
    """
    client = awsclient.get_client('cloudformation')

    status = None
    while status not in ['CREATE_COMPLETE', 'FAILED']:
        response = client.describe_change_set(
            ChangeSetName=change_set_name,
            StackName=stack_name)
        status = response['Status']
        # print('##### %s' % status)
        if status == 'FAILED':
            print(response['StatusReason'])
        elif status == 'CREATE_COMPLETE':
            for change in response['Changes']:
                print(json2table(change['ResourceChange']))