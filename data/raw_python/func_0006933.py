def down_ec2(instance_id, region, access_key_id, secret_access_key):
    """ shutdown of an existing EC2 instance """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)
    # get the instance_id from the state file, and stop the instance
    instance = conn.stop_instances(instance_ids=instance_id)[0]
    while instance.state != "stopped":
        log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    log_green('Instance state: %s' % instance.state)