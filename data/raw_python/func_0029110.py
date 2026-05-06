def output_deployment_summary(awsclient, deployment_id):
    """summary

    :param awsclient:
    :param deployment_id:
    """
    log.info('\ndeployment summary:')
    log.info('%-22s %-12s %s', 'Instance ID', 'Status', 'Most recent event')
    for instance_id in _list_deployment_instances(awsclient, deployment_id):
        status, last_event = \
            _get_deployment_instance_summary(awsclient, deployment_id, instance_id)
        log.info(Fore.MAGENTA + '%-22s' + Fore.RESET + ' %-12s %s',
                 instance_id, status, last_event)