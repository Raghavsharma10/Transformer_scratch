def list_stacks(awsclient):
    """Print out the list of stacks deployed at AWS cloud.

    :param awsclient:
    :return:
    """
    client_cf = awsclient.get_client('cloudformation')
    response = client_cf.list_stacks(
        StackStatusFilter=[
            'CREATE_IN_PROGRESS', 'CREATE_COMPLETE', 'ROLLBACK_IN_PROGRESS',
            'ROLLBACK_COMPLETE', 'DELETE_IN_PROGRESS', 'DELETE_FAILED',
            'UPDATE_IN_PROGRESS', 'UPDATE_COMPLETE_CLEANUP_IN_PROGRESS',
            'UPDATE_COMPLETE', 'UPDATE_ROLLBACK_IN_PROGRESS',
            'UPDATE_ROLLBACK_FAILED',
            'UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS',
            'UPDATE_ROLLBACK_COMPLETE',
        ]
    )
    result = {}
    stack_sum = 0
    for summary in response['StackSummaries']:
        result['StackName'] = summary["StackName"]
        result['CreationTime'] = summary['CreationTime']
        result['StackStatus'] = summary['StackStatus']
        print(json2table(result))
        stack_sum += 1
    print('listed %s stacks' % str(stack_sum))