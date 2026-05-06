def stop_deployment(awsclient, deployment_id):
    """stop tenkai deployment.

    :param awsclient:
    :param deployment_id:
    """
    log.info('Deployment: %s - stopping active deployment.', deployment_id)
    client_codedeploy = awsclient.get_client('codedeploy')

    response = client_codedeploy.stop_deployment(
        deploymentId=deployment_id,
        autoRollbackEnabled=True
    )