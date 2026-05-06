def stop_stack(awsclient, stack_name, use_suspend=False):
    """Stop an existing stack on AWS cloud.

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
        client_ec2 = awsclient.get_client('ec2')

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

        for asg in autoscaling_groups:
            # find instances in autoscaling group
            ec2_instances = all_pages(
                client_autoscaling.describe_auto_scaling_instances,
                {},
                lambda r: [i['InstanceId'] for i in r.get('AutoScalingInstances', [])
                           if i['AutoScalingGroupName'] == asg['PhysicalResourceId']],
            )

            if use_suspend:
                # alternative implementation to speed up start
                # only problem is that instances must survive stop & start
                # suspend all autoscaling processes
                log.info('Suspending all autoscaling processes for \'%s\'',
                         asg['LogicalResourceId'])
                response = client_autoscaling.suspend_processes(
                    AutoScalingGroupName=asg['PhysicalResourceId'],
                    ScalingProcesses=scaling_process_types
                )

                _stop_ec2_instances(awsclient, ec2_instances)
            else:
                # resize autoscaling group (min, max = 0)
                log.info('Resize autoscaling group \'%s\' to minSize=0, maxSize=0',
                         asg['LogicalResourceId'])
                response = client_autoscaling.update_auto_scaling_group(
                    AutoScalingGroupName=asg['PhysicalResourceId'],
                    MinSize=0,
                    MaxSize=0
                )
                if ec2_instances:
                    running_instances = all_pages(
                        client_ec2.describe_instance_status,
                        {
                            'InstanceIds': ec2_instances,
                            'Filters': [{
                                'Name': 'instance-state-name',
                                'Values': ['pending', 'running']
                            }]
                        },
                        lambda r: [i['InstanceId'] for i in r.get('InstanceStatuses', [])],
                    )
                    if running_instances:
                        # wait for instances to terminate
                        waiter_inst_terminated = client_ec2.get_waiter('instance_terminated')
                        waiter_inst_terminated.wait(InstanceIds=running_instances)

        # setting ECS desiredCount to zero
        services = [
            r for r in resources
            if r['ResourceType'] == 'AWS::ECS::Service'
        ]
        if services:
            template, parameters = _get_template_parameters(awsclient, stack_name)
            _stop_ecs_services(awsclient, services, template, parameters)

        # stopping ec2 instances
        instances = [
            r['PhysicalResourceId'] for r in resources
            if r['ResourceType'] == 'AWS::EC2::Instance'
        ]
        _stop_ec2_instances(awsclient, instances)

        # stopping db instances
        db_instances = [
            r['PhysicalResourceId'] for r in resources
            if r['ResourceType'] == 'AWS::RDS::DBInstance'
        ]
        running_db_instances = _filter_db_instances_by_status(
            awsclient, db_instances, ['available']
        )
        for db in running_db_instances:
            log.info('Stopping RDS instance \'%s\'', db)
            client_rds.stop_db_instance(DBInstanceIdentifier=db)

    return exit_code