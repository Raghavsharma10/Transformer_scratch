def _stop_ecs_services(awsclient, services, template, parameters, wait=True):
    """Helper to change desiredCount of ECS services to zero.
    By default it waits for this to complete.
    Docs here: http://docs.aws.amazon.com/cli/latest/reference/ecs/update-service.html

    :param awsclient:
    :param services:
    :param template: the cloudformation template
    :param parameters: the parameters used for the cloudformation template
    :param wait: waits for services to stop
    :return:
    """
    if len(services) == 0:
        return
    client_ecs = awsclient.get_client('ecs')

    for service in services:
        log.info('Resize ECS service \'%s\' to desiredCount=0',
                 service['LogicalResourceId'])
        cluster, desired_count = _get_service_cluster_desired_count(
            template, parameters, service['LogicalResourceId'])
        log.debug('cluster: %s' % cluster)
        response = client_ecs.update_service(
            cluster=cluster,
            service=service['PhysicalResourceId'],
            desiredCount=0
        )