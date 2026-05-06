def _create_snapshot(volume):
    """ Create a new snapshot

    :type volume: boto.ec2.volume.Volume
    :param volume: Volume to snapshot
    :returns: boto.ec2.snapshot.Snapshot -- The new snapshot
    """
    logger.info('Creating new snapshot for {}'.format(volume.id))
    snapshot = volume.create_snapshot(
        description="Automatic snapshot by Automated EBS Snapshots")
    logger.info('Created snapshot {} for volume {}'.format(
        snapshot.id, volume.id))

    return snapshot