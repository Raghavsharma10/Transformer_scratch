def get_outputs_for_stack(awsclient, stack_name):
    """
    Read environment from ENV and mangle it to a (lower case) representation
    Note: gcdt.servicediscovery get_outputs_for_stack((awsclient, stack_name)
    is used in many cloudformation.py templates!

    :param awsclient:
    :param stack_name:
    :return: dictionary containing the stack outputs
    """
    client_cf = awsclient.get_client('cloudformation')
    response = client_cf.describe_stacks(StackName=stack_name)
    if response['Stacks'] and 'Outputs' in response['Stacks'][0]:
        result = {}
        for output in response['Stacks'][0]['Outputs']:
            result[output['OutputKey']] = output['OutputValue']
        return result