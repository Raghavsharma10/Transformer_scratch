def file_key_regenerate( blockchain_id, hostname, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Generate a new encryption key.
    Retire the existing key, if it exists.
    Return {'status': True} on success
    Return {'error': ...} on error
    """
    
    config_dir = os.path.dirname(config_path)
    current_key = file_key_lookup( blockchain_id, 0, hostname, config_path=config_path )
    if 'status' in current_key and current_key['status']:
        # retire
        # NOTE: implicitly depends on this method failing only because the key doesn't exist
        res = file_key_retire( blockchain_id, current_key, config_path=config_path, wallet_keys=wallet_keys )
        if 'error' in res:
            log.error("Failed to retire key %s: %s" % (current_key['key_id'], res['error']))
            return {'error': 'Failed to retire key'}

    # make a new key 
    res = blockstack_gpg.gpg_app_create_key( blockchain_id, "files", hostname, wallet_keys=wallet_keys, config_dir=config_dir )
    if 'error' in res:
        log.error("Failed to generate new key: %s" % res['error'])
        return {'error': 'Failed to generate new key'}

    return {'status': True}