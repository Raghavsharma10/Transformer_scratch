def dump_selected_keys_or_addrs(wallet_obj, used=None, zero_balance=None):
    '''
    Works for both public key only or private key access
    '''
    if wallet_obj.private_key:
        content_str = 'private keys'
    else:
        content_str = 'addresses'

    if not USER_ONLINE:
        puts(colored.red('\nInternet connection required, would you like to dump *all* %s instead?' % (
            content_str,
            content_str,
            )))
        if confirm(user_prompt=DEFAULT_PROMPT, default=True):
            dump_all_keys_or_addrs(wallet_obj=wallet_obj)
        else:
            return

    mpub = wallet_obj.serialize_b58(private=False)

    if wallet_obj.private_key is None:
        puts('Displaying Public Addresses Only')
        puts('For Private Keys, please open bcwallet with your Master Private Key:\n')
        priv_to_display = '%s123...' % first4mprv_from_mpub(mpub=mpub)

        print_bcwallet_basic_priv_opening(priv_to_display=priv_to_display)

    chain_address_objs = get_addresses_on_both_chains(
            wallet_obj=wallet_obj,
            used=used,
            zero_balance=zero_balance,
            )

    if wallet_obj.private_key and chain_address_objs:
        print_childprivkey_warning()

    addr_cnt = 0
    for chain_address_obj in chain_address_objs:
        if chain_address_obj['index'] == 0:
            print_external_chain()
        elif chain_address_obj['index'] == 1:
            print_internal_chain()
        print_key_path_header()
        for address_obj in chain_address_obj['chain_addresses']:

            print_path_info(
                    address=address_obj['pub_address'],
                    wif=address_obj.get('wif'),
                    path=address_obj['path'],
                    coin_symbol=coin_symbol_from_mkey(mpub),
                    )

            addr_cnt += 1

    if addr_cnt:
        puts(colored.blue('\nYou can compare this output to bip32.org'))
    else:
        puts('No matching %s in this subset. Would you like to dump *all* %s instead?' % (
            content_str,
            content_str,
            ))
        if confirm(user_prompt=DEFAULT_PROMPT, default=True):
            dump_all_keys_or_addrs(wallet_obj=wallet_obj)