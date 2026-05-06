def output_deployment_status(awsclient, deployment_id, iterations=100):
    """Wait until an deployment is in an steady state and output information.

    :param deployment_id:
    :param iterations:
    :return: exit_code
    """
    counter = 0
    steady_states = ['Succeeded', 'Failed', 'Stopped']
    client_codedeploy = awsclient.get_client('codedeploy')

    while counter <= iterations:
        response = client_codedeploy.get_deployment(deploymentId=deployment_id)
        status = response['deploymentInfo']['status']

        if status not in steady_states:
            log.info('Deployment: %s - State: %s' % (deployment_id, status))
            time.sleep(10)
        elif status == 'Failed':
            log.info(
                colored.red('Deployment: {} failed: {}'.format(
                    deployment_id,
                    json.dumps(response['deploymentInfo']['errorInformation'],
                               indent=2)
                ))
            )
            return 1
        else:
            log.info('Deployment: %s - State: %s' % (deployment_id, status))
            break

    return 0