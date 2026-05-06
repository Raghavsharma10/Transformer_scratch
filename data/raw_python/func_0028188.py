def list(connection):
    """ List watched EBS volumes

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :returns: None
    """
    volumes = get_watched_volumes(connection)

    if not volumes:
        logger.info('No watched volumes found')
        return

    logger.info(
        '+-----------------------'
        '+----------------------'
        '+--------------'
        '+------------+')
    logger.info(
        '| {volume:<21} '
        '| {volume_name:<20.20} '
        '| {interval:<12} '
        '| {retention:<10} |'.format(
            volume='Volume ID',
            volume_name='Volume name',
            interval='Interval',
            retention='Retention'))
    logger.info(
        '+-----------------------'
        '+----------------------'
        '+--------------'
        '+------------+')

    for volume in volumes:
        if 'AutomatedEBSSnapshots' not in volume.tags:
            interval = 'Interval tag not found'
        elif volume.tags['AutomatedEBSSnapshots'] not in VALID_INTERVALS:
            interval = 'Invalid interval'
        else:
            interval = volume.tags['AutomatedEBSSnapshots']

        if 'AutomatedEBSSnapshotsRetention' not in volume.tags:
            retention = 0
        else:
            retention = volume.tags['AutomatedEBSSnapshotsRetention']

        # Get the volume name
        try:
            volume_name = volume.tags['Name']
        except KeyError:
            volume_name = ''

        logger.info(
            '| {volume_id:<14} '
            '| {volume_name:<20.20} '
            '| {interval:<12} '
            '| {retention:<10} |'.format(
                volume_id=volume.id,
                volume_name=volume_name,
                interval=interval,
                retention=retention))

    logger.info(
        '+-----------------------'
        '+----------------------'
        '+--------------'
        '+------------+')