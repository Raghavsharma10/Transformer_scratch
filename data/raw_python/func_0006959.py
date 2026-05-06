def up_ec2(region,
           access_key_id,
           secret_access_key,
           instance_id,
           username):
    """ boots an existing ec2_instance """

    conn = connect_to_ec2(region, access_key_id, secret_access_key)
    # boot the ec2 instance
    instance = conn.start_instances(instance_ids=instance_id)[0]
    while instance.state != "running":
        log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    # the ip_address has changed so we need to get the latest data from ec2
    data = get_ec2_info(instance_id=instance_id,
                        region=region,
                        access_key_id=access_key_id,
                        secret_access_key=secret_access_key,
                        username=username)
    # and make sure we don't return until the instance is fully up
    wait_for_ssh(data['ip_address'])
    # lets update our local state file with the new ip_address
    save_ec2_state_locally(instance_id=instance_id,
                           region=region,
                           username=username,
                           access_key_id=access_key_id,
                           secret_access_key=secret_access_key)

    env.hosts = data['ip_address']

    print_ec2_info(region,
                   instance_id,
                   access_key_id,
                   secret_access_key,
                   username)