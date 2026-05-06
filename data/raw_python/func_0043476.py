def file_key_retire( blockchain_id, file_key, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Retire the given key.  Move it to the head of the old key bundle list
    @file_key should be data returned by file_key_lookup
    Return {'status': True} on success
    Return {'error': ...} on error
    """

    config_dir = os.path.dirname(config_path)
    url = file_url_expired_keys( blockchain_id )
    proxy = blockstack_client.get_default_proxy( config_path=config_path )
        
    old_key_bundle_res = blockstack_client.data_get( url, wallet_keys=wallet_keys, proxy=proxy )
    if 'error' in old_key_bundle_res:
        log.warn('Failed to get old key bundle: %s' % old_key_bundle_res['error'])
        old_key_list = []

    else:
        old_key_list = old_key_bundle_res['data']['old_keys']
        for old_key in old_key_list:
            if old_key['key_id'] == file_key['key_id']:
                # already present 
                log.warning("Key %s is already retired" % file_key['key_id'])
                return {'status': True}

    old_key_list.insert(0, file_key )

    res = blockstack_client.data_put( url, {'old_keys': old_key_list}, wallet_keys=wallet_keys, proxy=proxy )
    if 'error' in res:
        log.error("Failed to append to expired key bundle: %s" % res['error'])
        return {'error': 'Failed to append to expired key list'}

    return {'status': True}