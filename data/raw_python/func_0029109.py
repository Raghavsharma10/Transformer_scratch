def _get_deployment_instance_diagnostics(awsclient, deployment_id, instance_id):
    """Gets you the diagnostics details for the first 'Failed' event.

    :param awsclient:
    :param deployment_id:
    :param instance_id:
    return: None or (error_code, script_name, message, log_tail)
    """
    client_codedeploy = awsclient.get_client('codedeploy')
    request = {
        'deploymentId': deployment_id,
        'instanceId': instance_id
    }
    response = client_codedeploy.get_deployment_instance(**request)
    # find first 'Failed' event
    for i, event in enumerate(response['instanceSummary']['lifecycleEvents']):
        if event['status'] == 'Failed':
            return event['diagnostics']['errorCode'], \
                   event['diagnostics']['scriptName'], \
                   event['diagnostics']['message'], \
                   event['diagnostics']['logTail']
    return None