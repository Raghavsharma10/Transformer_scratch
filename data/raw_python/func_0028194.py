def list_snapshots(connection, volume):
    """ List all snapshots for the volume

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type volume: str
    :param volume: Volume ID or Volume Name
    :returns: None
    """

    logger.info(
        '+----------------'
        '+----------------------'
        '+---------------------------+')
    logger.info(
        '| {snapshot:<14} '
        '| {snapshot_name:<20.20} '
        '| {created:<25} |'.format(
            snapshot='Snapshot ID',
            snapshot_name='Snapshot name',
            created='Created'))
    logger.info(
        '+----------------'
        '+----------------------'
        '+---------------------------+')

    vid = get_volume_id(connection, volume)
    if vid:
        vol = connection.get_all_volumes(volume_ids=[vid])[0]
        for snap in vol.snapshots():
            logger.info(
                '| {snapshot:<14} '
                '| {snapshot_name:<20.20} '
                '| {created:<25} |'.format(
                    snapshot=snap.id,
                    snapshot_name=snap.tags.get('Name', ''),
                    created=snap.start_time))

    logger.info(
        '+----------------'
        '+----------------------'
        '+---------------------------+')