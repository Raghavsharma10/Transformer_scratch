def ebs_volume_exists(connection, region, volume_id):
    """ finds out if a ebs volume exists """
    for vol in connection.get_all_volumes():
        if vol.id == volume_id:
            return True
    return False