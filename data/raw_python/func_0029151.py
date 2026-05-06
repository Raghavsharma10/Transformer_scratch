def get_parameter_diff(awsclient, config):
    """get differences between local config and currently active config
    """
    client_cf = awsclient.get_client('cloudformation')
    try:
        stack_name = config['stack']['StackName']
        if stack_name:
            response = client_cf.describe_stacks(StackName=stack_name)
            if response['Stacks']:
                stack_id = response['Stacks'][0]['StackId']
                stack = response['Stacks'][0]
            else:
                return None
        else:
            print(
                'StackName is not configured, could not create parameter diff')
            return None
    except GracefulExit:
        raise
    except Exception:
        # probably the stack is not existent
        return None

    changed = 0
    table = []
    table.append(['Parameter', 'Current Value', 'New Value'])

    # Check if there are parameters for the stack
    if 'Parameters' in stack:
        for param in stack['Parameters']:
            try:
                old = str(param['ParameterValue'])
                # can not compare list with str!!
                # if ',' in old:
                #    old = old.split(',')
                new = config['parameters'][param['ParameterKey']]
                if old != new:
                    if old.startswith('***'):
                        # parameter is configured with `NoEcho=True`
                        # this means we can not really say if the value changed!!
                        # for security reasons we block viewing the new value
                        new = old
                    table.append([param['ParameterKey'], old, new])
                    changed += 1
            except GracefulExit:
                raise
            except Exception:
                print('Did not find %s in local config file' % param[
                    'ParameterKey'])

    if changed > 0:
        print(tabulate(table, tablefmt='fancy_grid'))

    return changed > 0