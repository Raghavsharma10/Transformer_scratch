def file_list( blockchain_id, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    List all files uploaded to a particular blockchain ID
    Return {'status': True, 'listing': list} on success
    Return {'error': ...} on error
    """

    config_dir = os.path.dirname(config_path)
    client_config_path = os.path.join(config_dir, blockstack_client.CONFIG_FILENAME )
    proxy = blockstack_client.get_default_proxy( config_path=client_config_path )

    res = blockstack_client.data_list( blockchain_id, wallet_keys=wallet_keys, proxy=proxy )
    if 'error' in res:
        log.error("Failed to list data: %s" % res['error'])
        return {'error': 'Failed to list data'}

    listing = []

    # find the ones that this app put there 
    for rec in res['listing']:
        if not file_is_fq_data_name( rec['data_id'] ):
            continue
        
        listing.append( rec )

    return {'status': True, 'listing': listing}