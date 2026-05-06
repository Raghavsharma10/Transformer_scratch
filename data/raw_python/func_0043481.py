def file_sign( blockchain_id, hostname, input_path, passphrase=None, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Sign a file with the current blockchain ID's host's public key.
    @config_path should be for the *client*, not blockstack-file
    Return {'status': True, 'sender_key_id': ..., 'sig': ...} on success, and write ciphertext to output_path
    Return {'error': ...} on error
    """
    config_dir = os.path.dirname(config_path)

    # find our encryption key
    key_info = file_key_lookup( blockchain_id, 0, hostname, config_path=config_path, wallet_keys=wallet_keys )
    if 'error' in key_info:
        return {'error': 'Failed to lookup encryption key'}

    # sign
    res = blockstack_gpg.gpg_sign( input_path, key_info, config_dir=config_dir )
    if 'error' in res:
        log.error("Failed to encrypt: %s" % res['error'])
        return {'error': 'Failed to encrypt'}

    return {'status': True, 'sender_key_id': key_info['key_id'], 'sig': res['sig']}