def _start_ec2_instances(awsclient, ec2_instances, wait=True):
    """Helper to start ec2 instances

    :param awsclient:
    :param ec2_instances:
    :param wait: waits for instances to start
    :return:
    """
    if len(ec2_instances) == 0:
        return
    client_ec2 = awsclient.get_client('ec2')

    # get stopped instances
    stopped_instances = all_pages(
        client_ec2.describe_instance_status,
        {
            'InstanceIds': ec2_instances,
            'Filters': [{
                'Name': 'instance-state-name',
                'Values': ['stopping', 'stopped']
            }],
            'IncludeAllInstances': True
        },
        lambda r: [i['InstanceId'] for i in r.get('InstanceStatuses', [])],
    )

    if stopped_instances:
        # start all stopped instances
        log.info('Starting EC2 instances: %s', stopped_instances)
        client_ec2.start_instances(InstanceIds=stopped_instances)

        if wait:
            # wait for instances to come up
            waiter_inst_running = client_ec2.get_waiter('instance_running')
            waiter_inst_running.wait(InstanceIds=stopped_instances)

            # wait for status checks
            waiter_status_ok = client_ec2.get_waiter('instance_status_ok')
            waiter_status_ok.wait(InstanceIds=stopped_instances)