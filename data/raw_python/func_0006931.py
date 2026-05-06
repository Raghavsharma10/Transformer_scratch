def destroy_ec2(region, instance_id, access_key_id, secret_access_key):
    """ terminates the instance """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)

    data = get_ec2_info(instance_id=instance_id,
                        region=region,
                        access_key_id=access_key_id,
                        secret_access_key=secret_access_key,
                        username=None)

    instance = conn.terminate_instances(instance_ids=[data['id']])[0]
    log_yellow('destroying instance ...')
    while instance.state != "terminated":
        log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    volume_id = data['volume']
    if volume_id:
        destroy_ebs_volume(region, volume_id, access_key_id,
                           secret_access_key)
    os.unlink('data.json')