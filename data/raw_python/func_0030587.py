def unify_mp(b, partition_name):
    """Unify all of the segment partitions for a parent partition, then run stats on the MPR file"""

    with b.progress.start('coalesce_mp',0,message="MP coalesce {}".format(partition_name)) as ps:
        r = b.unify_partition(partition_name, None, ps)

    return r