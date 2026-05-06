def dump_all_keys_or_addrs(wallet_obj):
    '''
    Offline-enabled mechanism to dump addresses
    '''

    print_traversal_warning()

    puts('\nDo you understand this warning?')
    if not confirm(user_prompt=DEFAULT_PROMPT, default=False):
        puts(colored.red('Dump Cancelled!'))
        return

    mpub = wallet_obj.serialize_b58(private=False)

    if wallet_obj.private_key:
        desc_str = 'private keys'
    else:
        desc_str = 'addresses'
        puts('Displaying Public Addresses Only')
        puts('For Private Keys, please open bcwallet with your Master Private Key:\n')
        priv_to_display = '%s123...' % first4mprv_from_mpub(mpub=mpub)
        print_bcwallet_basic_priv_opening(priv_to_display=priv_to_display)

    puts('How many %s (on each chain) do you want to dump?' % desc_str)
    puts('Enter "b" to go back.\n')

    num_keys = get_int(
            user_prompt=DEFAULT_PROMPT,
            max_int=10**5,
            default_input='5',
            show_default=True,
            quit_ok=True,
            )

    if num_keys is False:
        return

    if wallet_obj.private_key:
        print_childprivkey_warning()

    puts('-' * 70)
    for chain_int in (0, 1):
        for current in range(0, num_keys):
            path = "m/%d/%d" % (chain_int, current)
            if current == 0:
                if chain_int == 0:
                    print_external_chain()
                    print_key_path_header()
                elif chain_int == 1:
                    print_internal_chain()
                    print_key_path_header()
            child_wallet = wallet_obj.get_child_for_path(path)
            if wallet_obj.private_key:
                wif_to_use = child_wallet.export_to_wif()
            else:
                wif_to_use = None
            print_path_info(
                    address=child_wallet.to_address(),
                    path=path,
                    wif=wif_to_use,
                    coin_symbol=coin_symbol_from_mkey(mpub),
                    )

    puts(colored.blue('\nYou can compare this output to bip32.org'))