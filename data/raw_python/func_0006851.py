def destroy_ec2(connection, region, instance_id, log=False):
    """ terminates the instance """

    data = get_ec2_info(connection=connection,
                        instance_id=instance_id,
                        region=region)

    instance = connection.terminate_instances(instance_ids=[data['id']])[0]
    if log:
        log_yellow('destroying instance ...')
    while instance.state != "terminated":
        if log:
            log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    volume_id = data['volume']
    if volume_id:
        destroy_ebs_volume(connection, region, volume_id)