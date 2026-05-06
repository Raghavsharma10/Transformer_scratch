def ebs_volume_exists(region, volume_id, access_key_id, secret_access_key):
    """ finds out if a ebs volume exists """
    conn = connect_to_ec2(region, access_key_id, secret_access_key)
    for vol in conn.get_all_volumes():
        if vol.id == volume_id:
            return True