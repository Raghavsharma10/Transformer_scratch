def up_ec2(connection,
           region,
           instance_id,
           wait_for_ssh_available=True,
           log=False,
           timeout=600):
    """ boots an existing ec2_instance """

    # boot the ec2 instance
    instance = connection.start_instances(instance_ids=instance_id)[0]
    instance.update()
    while instance.state != "running" and timeout > 1:
        log_yellow("Instance state: %s" % instance.state)
        if log:
            log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        timeout = timeout - 10
        instance.update()

    # and make sure we don't return until the instance is fully up
    if wait_for_ssh_available:
        wait_for_ssh(instance.ip_address)