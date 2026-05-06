def file_encrypt( blockchain_id, hostname, recipient_blockchain_id_and_hosts, input_path, output_path, passphrase=None, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Encrypt a file for a set of recipients.
    @recipient_blockchain_id_and_hosts must contain a list of (blockchain_id, hostname)
    Return {'status': True, 'sender_key_id': ...} on success, and write ciphertext to output_path
    Return {'error': ...} on error
    """
    config_dir = os.path.dirname(config_path)

    # find our encryption key
    key_info = file_key_lookup( blockchain_id, 0, hostname, config_path=config_path, wallet_keys=wallet_keys )
    if 'error' in key_info:
        return {'error': 'Failed to lookup encryption key'}

    # find the encryption key IDs for the recipients 
    recipient_keys = []
    for (recipient_id, recipient_hostname) in recipient_blockchain_id_and_hosts:
        if recipient_id == blockchain_id and recipient_hostname == hostname:
            # already have it 
            recipient_keys.append(key_info)
            continue

        recipient_info = file_key_lookup( recipient_id, 0, recipient_hostname, config_path=config_path, wallet_keys=wallet_keys )
        if 'error' in recipient_info:
            return {'error': "Failed to look up key for '%s'" % recipient_id}

        recipient_keys.append(recipient_info)

    # encrypt
    res = None
    with open(input_path, "r") as f:
        res = blockstack_gpg.gpg_encrypt( f, output_path, key_info, recipient_keys, passphrase=passphrase, config_dir=config_dir )
        
    if 'error' in res:
        log.error("Failed to encrypt: %s" % res['error'])
        return {'error': 'Failed to encrypt'}

    return {'status': True, 'sender_key_id': key_info['key_id']}