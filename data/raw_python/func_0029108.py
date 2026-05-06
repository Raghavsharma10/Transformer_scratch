def _get_deployment_instance_summary(awsclient, deployment_id, instance_id):
    """instance summary.

    :param awsclient:
    :param deployment_id:
    :param instance_id:
    return: status, last_event
    """
    client_codedeploy = awsclient.get_client('codedeploy')
    request = {
        'deploymentId': deployment_id,
        'instanceId': instance_id
    }
    response = client_codedeploy.get_deployment_instance(**request)
    return response['instanceSummary']['status'], \
           response['instanceSummary']['lifecycleEvents'][-1]['lifecycleEventName']