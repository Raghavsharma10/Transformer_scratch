def _stop_ec2_instances(awsclient, ec2_instances, wait=True):
    """Helper to stop ec2 instances.
    By default it waits for instances to stop.

    :param awsclient:
    :param ec2_instances:
    :param wait: waits for instances to stop
    :return:
    """
    if len(ec2_instances) == 0:
        return
    client_ec2 = awsclient.get_client('ec2')

    # get running instances
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
        log.info('Stopping EC2 instances: %s', running_instances)
        client_ec2.stop_instances(InstanceIds=running_instances)

        if wait:
            # wait for instances to stop
            waiter_inst_stopped = client_ec2.get_waiter('instance_stopped')
            waiter_inst_stopped.wait(InstanceIds=running_instances)