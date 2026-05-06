def create_image(cloud, **kwargs):
    """ proxy call for ec2, rackspace create ami backend functions """
    if cloud == 'ec2':
        return create_ami(**kwargs)

    if cloud == 'rackspace':
        return create_rackspace_image(**kwargs)

    if cloud == 'gce':
        return create_gce_image(**kwargs)