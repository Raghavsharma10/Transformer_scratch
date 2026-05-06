def _remove_old_snapshots(connection, volume):
    """ Remove old snapshots

    :type connection: boto.ec2.connection.EC2Connection
    :param connection: EC2 connection object
    :type volume: boto.ec2.volume.Volume
    :param volume: Volume to check
    :returns: None
    """
    if 'AutomatedEBSSnapshotsRetention' not in volume.tags:
        logger.warning(
            'Missing tag AutomatedEBSSnapshotsRetention for volume {}'.format(
                volume.id))
        return
    retention = int(volume.tags['AutomatedEBSSnapshotsRetention'])

    snapshots = connection.get_all_snapshots(filters={'volume-id': volume.id})

    # Sort the list based on the start time
    snapshots.sort(key=lambda x: x.start_time)

    # Remove snapshots we want to keep
    snapshots = snapshots[:-int(retention)]

    if not snapshots:
        logger.info('No old snapshots to remove')
        return

    for snapshot in snapshots:
        logger.info('Deleting snapshot {}'.format(snapshot.id))
        try:
            snapshot.delete()
        except EC2ResponseError as error:
            logger.warning('Could not remove snapshot: {}'.format(
                error.message))

    logger.info('Done deleting snapshots')