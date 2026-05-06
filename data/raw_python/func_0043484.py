def file_delete( blockchain_id, data_name, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Remove a file
    Return {'status': True} on success
    Return {'error': error} on failure
    """

    config_dir = os.path.dirname(config_path)
    client_config_path = os.path.join(config_dir, blockstack_client.CONFIG_FILENAME )
    proxy = blockstack_client.get_default_proxy( config_path=client_config_path )

    fq_data_name = file_fq_data_name( data_name ) 
    res = blockstack_client.data_delete( blockstack_client.make_mutable_data_url( blockchain_id, fq_data_name, None ), proxy=proxy, wallet_keys=wallet_keys )
    if 'error' in res:
        log.error("Failed to delete: %s" % res['error'])
        return {'error': 'Failed to delete'}

    return {'status': True}