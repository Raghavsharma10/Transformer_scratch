def create_server_ec2(connection,
                      region,
                      disk_name,
                      disk_size,
                      ami,
                      key_pair,
                      instance_type,
                      tags={},
                      security_groups=None,
                      delete_on_termination=True,
                      log=False,
                      wait_for_ssh_available=True):
    """
    Creates EC2 Instance
    """

    if log:
        log_green("Started...")
        log_yellow("...Creating EC2 instance...")

    ebs_volume = EBSBlockDeviceType()
    ebs_volume.size = disk_size
    bdm = BlockDeviceMapping()
    bdm[disk_name] = ebs_volume

    # get an ec2 ami image object with our choosen ami
    image = connection.get_all_images(ami)[0]
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
        if log:
            log_yellow("Instance state: %s" % instance.state)
        sleep(10)
        instance.update()
    if log:
        log_green("Instance state: %s" % instance.state)
    if wait_for_ssh_available:
        wait_for_ssh(instance.public_dns_name)

    # update the EBS volumes to be deleted on instance termination
    if delete_on_termination:
        for dev, bd in instance.block_device_mapping.items():
            instance.modify_attribute('BlockDeviceMapping',
                                      ["%s=%d" % (dev, 1)])

    # add a tag to our instance
    connection.create_tags([instance.id], tags)

    if log:
        log_green("Public dns: %s" % instance.public_dns_name)

    # returns our new instance
    return instance