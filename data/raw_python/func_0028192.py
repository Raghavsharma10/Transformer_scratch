def watch_from_file(connection, file_name):
    """ Start watching a new volume

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type file_name: str
    :param file_name: path to config file
    :returns: None
    """
    with open(file_name, 'r') as filehandle:
        for line in filehandle.xreadlines():
            volume, interval, retention = line.rstrip().split(',')
            watch(
                connection,
                get_volume_id(connection, volume),
                interval, retention)