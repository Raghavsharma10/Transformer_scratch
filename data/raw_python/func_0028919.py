def start_stack(awsclient, stack_name, use_suspend=False):
    """Start an existing stack on AWS cloud.

    :param awsclient:
    :param stack_name:
    :param use_suspend: use suspend and resume on the autoscaling group
    :return: exit_code
    """
    exit_code = 0

    # check for DisableStop
    #disable_stop = conf.get('deployment', {}).get('DisableStop', False)
    #if disable_stop:
    #    log.warn('\'DisableStop\' is set - nothing to do!')
    #else:
    if not stack_exists(awsclient, stack_name):
        log.warn('Stack \'%s\' not deployed - nothing to do!', stack_name)
    else:
        client_cfn = awsclient.get_client('cloudformation')
        client_autoscaling = awsclient.get_client('autoscaling')
        client_rds = awsclient.get_client('rds')

        resources = all_pages(
            client_cfn.list_stack_resources,
            { 'StackName': stack_name },
            lambda r: r['StackResourceSummaries']
        )

        autoscaling_groups = [
            r for r in resources
            if r['ResourceType'] == 'AWS::AutoScaling::AutoScalingGroup'
        ]

        # lookup all types of scaling processes
        #    [Launch, Terminate, HealthCheck, ReplaceUnhealthy, AZRebalance
        #     AlarmNotification, ScheduledActions, AddToLoadBalancer]
        response = client_autoscaling.describe_scaling_process_types()
        scaling_process_types = [t['ProcessName'] for t in response.get('Processes', [])]

        # starting db instances
        db_instances = [
            r['PhysicalResourceId'] for r in resources
            if r['ResourceType'] == 'AWS::RDS::DBInstance'
        ]
        stopped_db_instances = _filter_db_instances_by_status(
            awsclient, db_instances, ['stopped']
        )
        for db in stopped_db_instances:
            log.info('Starting RDS instance \'%s\'', db)
            client_rds.start_db_instance(DBInstanceIdentifier=db)

        # wait for db instances to become available
        for db in stopped_db_instances:
            waiter_db_available = client_rds.get_waiter('db_instance_available')
            waiter_db_available.wait(DBInstanceIdentifier=db)

        # starting ec2 instances
        instances = [
            r['PhysicalResourceId'] for r in resources
            if r['ResourceType'] == 'AWS::EC2::Instance'
        ]
        _start_ec2_instances(awsclient, instances)

        services = [
            r for r in resources
            if r['ResourceType'] == 'AWS::ECS::Service'
        ]

        if (autoscaling_groups and not use_suspend) or services:
            template, parameters = _get_template_parameters(awsclient, stack_name)

        # setting ECS desiredCount back
        if services:
            _start_ecs_services(awsclient, services, template, parameters)

        for asg in autoscaling_groups:
            if use_suspend:
                # alternative implementation to speed up start
                # only problem is that instances must survive stop & start
                # find instances in autoscaling group
                instances = all_pages(
                    client_autoscaling.describe_auto_scaling_instances,
                    {},
                    lambda r: [i['InstanceId'] for i in r.get('AutoScalingInstances', [])
                               if i['AutoScalingGroupName'] == asg['PhysicalResourceId']],
                )
                _start_ec2_instances(awsclient, instances)

                # resume all autoscaling processes
                log.info('Resuming all autoscaling processes for \'%s\'',
                         asg['LogicalResourceId'])
                response = client_autoscaling.resume_processes(
                    AutoScalingGroupName=asg['PhysicalResourceId'],
                    ScalingProcesses=scaling_process_types
                )
            else:
                # resize autoscaling group back to its original values
                log.info('Resize autoscaling group \'%s\' back to original values',
                         asg['LogicalResourceId'])
                min, max = _get_autoscaling_min_max(
                    template, parameters, asg['LogicalResourceId'])
                response = client_autoscaling.update_auto_scaling_group(
                    AutoScalingGroupName=asg['PhysicalResourceId'],
                    MinSize=min,
                    MaxSize=max
                )

    return exit_code