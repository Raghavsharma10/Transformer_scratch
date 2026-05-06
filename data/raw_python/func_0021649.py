def get_addresses_on_both_chains(wallet_obj, used=None, zero_balance=None):
    '''
    Get addresses across both subchains based on the filter criteria passed in

    Returns a list of dicts of the following form:
        [
            {'address': '1abc123...', 'path': 'm/0/9', 'pubkeyhex': '0123456...'},
            ...,
        ]

    Dicts may also contain WIF and privkeyhex if wallet_obj has private key
    '''
    mpub = wallet_obj.serialize_b58(private=False)

    wallet_name = get_blockcypher_walletname_from_mpub(
            mpub=mpub,
            subchain_indices=[0, 1],
            )

    wallet_addresses = get_wallet_addresses(
            wallet_name=wallet_name,
            api_key=BLOCKCYPHER_API_KEY,
            is_hd_wallet=True,
            used=used,
            zero_balance=zero_balance,
            coin_symbol=coin_symbol_from_mkey(mpub),
            )
    verbose_print('wallet_addresses:')
    verbose_print(wallet_addresses)

    if wallet_obj.private_key:
        master_key = wallet_obj.serialize_b58(private=True)
    else:
        master_key = mpub

    chains_address_paths_cleaned = []
    for chain in wallet_addresses['chains']:
        if chain['chain_addresses']:
            chain_address_paths = verify_and_fill_address_paths_from_bip32key(
                    address_paths=chain['chain_addresses'],
                    master_key=master_key,
                    network=guess_network_from_mkey(mpub),
                    )
            chain_address_paths_cleaned = {
                    'index': chain['index'],
                    'chain_addresses': chain_address_paths,
                    }
            chains_address_paths_cleaned.append(chain_address_paths_cleaned)

    return chains_address_paths_cleaned