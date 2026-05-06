def file_decrypt( blockchain_id, hostname, sender_blockchain_id, sender_key_id, input_path, output_path, passphrase=None, config_path=CONFIG_PATH, wallet_keys=None ):
    """
    Decrypt a file from a sender's blockchain ID.
    Try our current key, and then the old keys
    (but warn if there are revoked keys)
    Return {'status': True} on success, and write plaintext to output_path
    Return {'error': ...} on failure
    """
    config_dir = os.path.dirname(config_path)
    decrypted = False
    old_key = False
    old_key_index = 0
    sender_old_key_index = 0

    # get the sender key 
    sender_key_info = file_key_lookup( sender_blockchain_id, None, None, key_id=sender_key_id, config_path=config_path, wallet_keys=wallet_keys ) 
    if 'error' in sender_key_info:
        log.error("Failed to look up sender key: %s" % sender_key_info['error'])
        return {'error': 'Failed to lookup sender key'}

    if 'stale_key_index' in sender_key_info.keys():
        old_key = True
        sender_old_key_index = sender_key_info['sender_key_index']

    # try each of our keys
    # current key...
    key_info = file_key_lookup( blockchain_id, 0, hostname, config_path=config_path, wallet_keys=wallet_keys )
    if 'error' not in key_info:
        res = file_decrypt_from_key_info( sender_key_info, blockchain_id, 0, hostname, input_path, output_path, passphrase=passphrase, config_path=config_path, wallet_keys=wallet_keys )
        if 'error' in res:
            if not res['status']:
                # permanent failure 
                log.error("Failed to decrypt: %s" % res['error'])
                return {'error': 'Failed to decrypt'}

        else:
            decrypted = True

    else:
        # did not look up key 
        log.error("Failed to lookup key: %s" % key_info['error'])

    if not decrypted:
        # try old keys 
        for i in xrange(1, MAX_EXPIRED_KEYS):
            res = file_decrypt_from_key_info( sender_key_info, blockchain_id, i, hostname, input_path, output_path, passphrase=passphrase, config_path=config_path, wallet_keys=wallet_keys )
            if 'error' in res:
                # key is not online, but don't try again 
                log.error("Failed to decrypt: %s" % res['error'])
                return {'error': 'Failed to decrypt'}
            else:
                decrypted = True
                old_key = True
                old_key_index = i
                break

    if decrypted:
        log.debug("Decrypted with %s.%s" % (blockchain_id, hostname))

        ret = {'status': True}
        if old_key:
            ret['warning'] = "Used stale key"
            ret['stale_key_index'] = old_key_index
            ret['stale_sender_key_index'] = sender_old_key_index

        return ret

    else:
        return {'error': 'No keys could decrypt'}