def destroy_ebs_volume(connection, region, volume_id, log=False):
    """ destroys an ebs volume """

    if ebs_volume_exists(connection, region, volume_id):
        if log:
            log_yellow('destroying EBS volume ...')
        try:
            connection.delete_volume(volume_id)
        except:
            # our EBS volume may be gone, but AWS info tables are stale
            # wait a bit and ask again
            sleep(5)
            if not ebs_volume_exists(connection, region, volume_id):
                pass
            else:
                raise("Couldn't delete EBS volume")