def file_key_lookup( blockchain_id, index, hostname, key_id=None, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Get the file-encryption GPG key for the given blockchain ID, by index.

    if index == 0, then give back the current key
    if index > 0, then give back an older (revoked) key.
    if key_id is given, index and hostname will be ignored

    Return {'status': True, 'key_data': ..., 'key_id': key_id, OPTIONAL['stale_key_index': idx]} on success
    Return {'error': ...} on failure
    """

    log.debug("lookup '%s' key for %s (index %s, key_id = %s)" % (hostname, blockchain_id, index, key_id))
    conf = get_config( config_path )
    config_dir = os.path.dirname(config_path)
    
    proxy = blockstack_client.get_default_proxy( config_path=config_path )
    immutable = conf['immutable_key']

    if key_id is not None:
        # we know exactly which key to get 
        # try each current key 
        hosts_listing = file_list_hosts( blockchain_id, wallet_keys=wallet_keys, config_path=config_path )
        if 'error' in hosts_listing:
            log.error("Failed to list hosts for %s: %s" % (blockchain_id, hosts_listing['error']))
            return {'error': 'Failed to look up hosts'}

        hosts = hosts_listing['hosts']
        for hostname in hosts:
            file_key = blockstack_gpg.gpg_app_get_key( blockchain_id, APP_NAME, hostname, immutable=immutable, key_id=key_id, config_dir=config_dir )
            if 'error' not in file_key:
                if key_id == file_key['key_id']:
                    # success!
                    return file_key

        # check previous keys...
        url = file_url_expired_keys( blockchain_id )
        old_key_bundle_res = blockstack_client.data_get( url, wallet_keys=wallet_keys, proxy=proxy )
        if 'error' in old_key_bundle_res:
            return old_key_bundle_res

        old_key_list = old_key_bundle_res['data']['old_keys']
        for i in xrange(0, len(old_key_list)):
            old_key = old_key_list[i]
            if old_key['key_id'] == key_id:
                # success!
                ret = {}
                ret.update( old_key )
                ret['stale_key_index'] = i+1 
                return old_key

        return {'error': 'No such key %s' % key_id}

    elif index == 0:
        file_key = blockstack_gpg.gpg_app_get_key( blockchain_id, APP_NAME, hostname, immutable=immutable, key_id=key_id, config_dir=config_dir )
        if 'error' in file_key:
            return file_key

        return file_key
    
    else:
        # get the bundle of revoked keys
        url = file_url_expired_keys( blockchain_id )
        old_key_bundle_res = blockstack_client.data_get( url, wallet_keys=wallet_keys, proxy=proxy )
        if 'error' in old_key_bundle_res:
            return old_key_bundle_res

        old_key_list = old_key_bundle_res['data']['old_keys']
        if index >= len(old_key_list)+1:
            return {'error': 'Index out of bounds: %s' % index}

        return old_key_list[index-1]