def _list_deployment_instances(awsclient, deployment_id):
    """list deployment instances.

    :param awsclient:
    :param deployment_id:
    """
    client_codedeploy = awsclient.get_client('codedeploy')

    instances = []
    next_token = None

    # TODO refactor generic exhaust_function from this
    while True:
        request = {
            'deploymentId': deployment_id
        }
        if next_token:
            request['nextToken'] = next_token
        response = client_codedeploy.list_deployment_instances(**request)
        instances.extend(response['instancesList'])
        if 'nextToken' not in response:
            break
        next_token = response['nextToken']
    return instances