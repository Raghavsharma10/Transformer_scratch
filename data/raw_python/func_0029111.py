def output_deployment_diagnostics(awsclient, deployment_id, log_group, start_time=None):
    """diagnostics

    :param awsclient:
    :param deployment_id:
    """
    headline = False
    for instance_id in _list_deployment_instances(awsclient, deployment_id):
        diagnostics = _get_deployment_instance_diagnostics(
            awsclient, deployment_id, instance_id)
        #if error_code != 'Success':
        if diagnostics is not None:
            error_code, script_name, message, log_tail = diagnostics
            # header
            if not headline:
                headline = True
                log.info('\ndeployment diagnostics:')
            # event logs
            log.info('Instance ID: %s', Fore.MAGENTA + instance_id + Fore.RESET)
            log.info('Error Code:  %s', error_code)
            log.info('Script Name: %s', script_name)
            log.info('Message:     %s', message)
            log.info('Log Tail:    %s', log_tail)
            # cloudwatch logs
            if check_log_stream_exists(awsclient, log_group, instance_id):
                logentries = get_log_events(
                    awsclient, log_group, instance_id,
                    datetime_to_timestamp(start_time))
                if logentries:
                    log.info('instance %s logentries', instance_id)
                    for e in logentries:
                        log.info(e['message'].strip())