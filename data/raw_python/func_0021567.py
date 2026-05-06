def verify_and_fill_address_paths_from_bip32key(address_paths, master_key, network):
    '''
    Take address paths and verifies their accuracy client-side.

    Also fills in all the available metadata (WIF, public key, etc)
    '''

    assert network, network

    wallet_obj = Wallet.deserialize(master_key, network=network)

    address_paths_cleaned = []

    for address_path in address_paths:
        path = address_path['path']
        input_address = address_path['address']
        child_wallet = wallet_obj.get_child_for_path(path)

        if child_wallet.to_address() != input_address:
            err_msg = 'Client Side Verification Fail for %s on %s:\n%s != %s' % (
                    path,
                    master_key,
                    child_wallet.to_address(),
                    input_address,
                    )
            raise Exception(err_msg)

        pubkeyhex = child_wallet.get_public_key_hex(compressed=True)

        server_pubkeyhex = address_path.get('public')
        if server_pubkeyhex and server_pubkeyhex != pubkeyhex:
            err_msg = 'Client Side Verification Fail for %s on %s:\n%s != %s' % (
                    path,
                    master_key,
                    pubkeyhex,
                    server_pubkeyhex,
                    )
            raise Exception(err_msg)

        address_path_cleaned = {
            'pub_address': input_address,
            'path': path,
            'pubkeyhex': pubkeyhex,
            }

        if child_wallet.private_key:
            privkeyhex = child_wallet.get_private_key_hex()
            address_path_cleaned['wif'] = child_wallet.export_to_wif()
            address_path_cleaned['privkeyhex'] = privkeyhex
        address_paths_cleaned.append(address_path_cleaned)

    return address_paths_cleaned