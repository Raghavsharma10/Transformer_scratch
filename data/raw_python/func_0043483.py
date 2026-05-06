def file_get( blockchain_id, hostname, sender_blockchain_id, data_name, output_path, passphrase=None, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Get a file from a known sender.
    Store it to output_path
    Return {'status': True} on success
    Return {'error': error} on failure
    """
  
    config_dir = os.path.dirname(config_path)
    client_config_path = os.path.join(config_dir, blockstack_client.CONFIG_FILENAME )
    proxy = blockstack_client.get_default_proxy( config_path=client_config_path )

    # get the ciphertext
    fq_data_name = file_fq_data_name( data_name ) 
    res = blockstack_client.data_get( blockstack_client.make_mutable_data_url( sender_blockchain_id, fq_data_name, None ), wallet_keys=wallet_keys, proxy=proxy )
    if 'error' in res:
        log.error("Failed to get ciphertext for %s: %s" % (fq_data_name, res['error']))
        return {'error': 'Failed to get encrypted file'}

    # stash
    fd, path = tempfile.mkstemp( prefix="blockstack-file-" )
    f = os.fdopen(fd, "w")
    f.write( res['data']['ciphertext'] )
    f.flush()
    os.fsync(f.fileno())
    f.close()

    sender_key_id = res['data']['sender_key_id']

    # decrypt it
    res = file_decrypt( blockchain_id, hostname, sender_blockchain_id, sender_key_id, path, output_path, passphrase=passphrase, config_path=config_path, wallet_keys=wallet_keys )
    os.unlink( path )
    if 'error' in res:
        log.error("Failed to decrypt: %s" % res['error'])
        return {'error': 'Failed to decrypt data'}

    else:
        # success!
        return res