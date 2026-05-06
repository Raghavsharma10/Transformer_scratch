def create_server(cloud, **kwargs):
    """
        Create a new instance
    """
    if cloud == 'ec2':
        _create_server_ec2(**kwargs)
    elif cloud == 'rackspace':
        _create_server_rackspace(**kwargs)
    elif cloud == 'gce':
        _create_server_gce(**kwargs)
    else:
        raise ValueError("Unknown cloud type: {}".format(cloud))