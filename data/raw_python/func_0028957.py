def get_lambdas(awsclient, config, add_arn=False):
    """Get the list of lambda functions.

    :param config:
    :param add_arn:
    :return: list containing lambda entries
    """
    if 'lambda' in config:
        client_lambda = awsclient.get_client('lambda')
        lambda_entries = config['lambda'].get('entries', [])
        lmbdas = []
        for lambda_entry in lambda_entries:
            lmbda = {
                'name': lambda_entry.get('name', None),
                'alias': lambda_entry.get('alias', None),
                'swagger_ref': lambda_entry.get('swaggerRef', None)
            }
            if add_arn:
                _sleep()
                response_lambda = client_lambda.get_function(
                    FunctionName=lmbda['name'])
                lmbda['arn'] = response_lambda['Configuration']['FunctionArn']
            lmbdas.append(lmbda)
        return lmbdas
    else:
        return []