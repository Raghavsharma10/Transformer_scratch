def destroy_ebs_volume(region, volume_id, access_key_id, secret_access_key):
    """ destroys an ebs volume """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)

    if ebs_volume_exists(region, volume_id, access_key_id, secret_access_key):
        log_yellow('destroying EBS volume ...')
        conn.delete_volume(volume_id)