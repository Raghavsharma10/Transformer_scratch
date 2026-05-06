def run(connection):
    """ Ensure that we have snapshots for a given volume

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :returns: None
    """
    volumes = volume_manager.get_watched_volumes(connection)

    for volume in volumes:
        _ensure_snapshot(connection, volume)
        _remove_old_snapshots(connection, volume)