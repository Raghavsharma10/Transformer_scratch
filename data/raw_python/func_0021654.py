def wallet_home(wallet_obj):
    '''
    Loaded on bootup (and stays in while loop until quitting)
    '''
    mpub = wallet_obj.serialize_b58(private=False)

    if wallet_obj.private_key is None:
        print_pubwallet_notice(mpub=mpub)
    else:
        print_bcwallet_basic_pub_opening(mpub=mpub)

    coin_symbol = coin_symbol_from_mkey(mpub)
    if USER_ONLINE:
        wallet_name = get_blockcypher_walletname_from_mpub(
                mpub=mpub,
                subchain_indices=[0, 1],
                )

        # Instruct blockcypher to track the wallet by pubkey
        create_hd_wallet(
                wallet_name=wallet_name,
                xpubkey=mpub,
                api_key=BLOCKCYPHER_API_KEY,
                coin_symbol=coin_symbol,
                subchain_indices=[0, 1],  # for internal and change addresses
                )

        # Display balance info
        display_balance_info(wallet_obj=wallet_obj)

    # Go to home screen
    while True:
        puts('-' * 70 + '\n')

        if coin_symbol in ('bcy', 'btc-testnet'):
            display_shortname = COIN_SYMBOL_MAPPINGS[coin_symbol]['display_shortname']
            if coin_symbol == 'bcy':
                faucet_url = 'https://accounts.blockcypher.com/blockcypher-faucet'
            elif coin_symbol == 'btc-testnet':
                faucet_url = 'https://accounts.blockcypher.com/testnet-faucet'
            puts('Get free %s faucet coins:' % display_shortname)
            puts(colored.blue(faucet_url))
            puts()

            if coin_symbol == 'btc-testnet':
                puts('Please consider returning unused testnet coins to mwmabpJVisvti3WEP5vhFRtn3yqHRD9KNP so we can distribute them to others.\n')

        puts('What do you want to do?:')
        if not USER_ONLINE:
            puts("(since you are NOT connected to BlockCypher, many choices are disabled)")
        with indent(2):
            puts(colored.cyan('1: Show balance and transactions'))
            puts(colored.cyan('2: Show new receiving addresses'))
            puts(colored.cyan('3: Send funds (more options here)'))

        with indent(2):
            if wallet_obj.private_key:
                puts(colored.cyan('0: Dump private keys and addresses (advanced users only)'))
            else:
                puts(colored.cyan('0: Dump addresses (advanced users only)'))

            puts(colored.cyan('\nq: Quit bcwallet\n'))

        choice = choice_prompt(
                user_prompt=DEFAULT_PROMPT,
                acceptable_responses=range(0, 3+1),
                quit_ok=True,
                default_input='1',
                )
        verbose_print('Choice: %s' % choice)

        if choice is False:
            puts(colored.green('Thanks for using bcwallet!'))
            print_keys_not_saved()
            break
        elif choice == '1':
            display_recent_txs(wallet_obj=wallet_obj)
        elif choice == '2':
            display_new_receiving_addresses(wallet_obj=wallet_obj)
        elif choice == '3':
            send_chooser(wallet_obj=wallet_obj)
        elif choice == '0':
            dump_private_keys_or_addrs_chooser(wallet_obj=wallet_obj)