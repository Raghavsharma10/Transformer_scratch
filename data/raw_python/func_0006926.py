def _create_server_ec2(region,
                       access_key_id,
                       secret_access_key,
                       disk_name,
                       disk_size,
                       ami,
                       key_pair,
                       instance_type,
                       username,
                       tags={},
                       security_groups=None):
    """
    Creates EC2 Instance and saves it state in a local json file
    """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)

    log_green("Started...")
    log_yellow("...Creating EC2 instance...")

    # we need a larger boot device to store our cached images
    ebs_volume = EBSBlockDeviceType()
    ebs_volume.size = disk_size
    bdm = BlockDeviceMapping()
    bdm[disk_name] = ebs_volume

    # get an ec2 ami image object with our choosen ami
    image = conn.get_all_images(ami)[0]
    # start a new instance
    reservation = image.run(1, 1,
                            key_name=key_pair,
                            security_groups=security_groups,
                            block_device_map=bdm,
                            instance_type=instance_type)

    # and get our instance_id
    instance = reservation.instances[0]

    #  and loop and wait until ssh is available
    while instance.state == u'pending':
        log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    log_green("Instance state: %s" % instance.state)
    wait_for_ssh(instance.public_dns_name)

    # update the EBS volumes to be deleted on instance termination
    for dev, bd in instance.block_device_mapping.items():
        instance.modify_attribute('BlockDeviceMapping',
                                  ["%s=%d" % (dev, 1)])

    # add a tag to our instance
    conn.create_tags([instance.id], tags)

    log_green("Public dns: %s" % instance.public_dns_name)
    # finally save the details or our new instance into the local state file
    save_ec2_state_locally(instance_id=instance.id,
                           region=region,
                           username=username,
                           access_key_id=access_key_id,
                           secret_access_key=secret_access_key)