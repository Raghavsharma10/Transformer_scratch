def watch(connection, volume_id, interval='daily', retention=0):
    """ Start watching a new volume

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type volume_id: str
    :param volume_id: VolumeID to add to the watchlist
    :type interval: str
    :param interval: Backup interval [hourly|daily|weekly|monthly|yearly]
    :type retention: int
    :param retention: Number of snapshots to keep. 0 == keep all
    :returns: bool - True if the watch was successful
    """
    try:
        volume = connection.get_all_volumes(volume_ids=[volume_id])[0]
    except EC2ResponseError:
        logger.warning('Volume {} not found'.format(volume_id))
        return False

    if interval not in VALID_INTERVALS:
        logger.warning(
            '{} is not a valid interval. Valid intervals are {}'.format(
                interval, ', '.join(VALID_INTERVALS)))

    # Remove the tag first
    volume.remove_tag('AutomatedEBSSnapshots')

    # Re-add the tag
    volume.add_tag('AutomatedEBSSnapshots', value=interval)

    # Remove the tag first
    volume.remove_tag('AutomatedEBSSnapshotsRetention')

    # Re-add the tag
    volume.add_tag('AutomatedEBSSnapshotsRetention', value=int(retention))

    logger.info('Updated the rotation interval to {} for {}'.format(
        interval, volume_id))

    return True