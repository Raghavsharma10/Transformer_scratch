def dump_private_keys_or_addrs_chooser(wallet_obj):
    '''
    Offline-enabled mechanism to dump everything
    '''

    if wallet_obj.private_key:
        puts('Which private keys and addresses do you want?')
    else:
        puts('Which addresses do you want?')
    with indent(2):
        puts(colored.cyan('1: Active - have funds to spend'))
        puts(colored.cyan('2: Spent - no funds to spend (because they have been spent)'))
        puts(colored.cyan('3: Unused - no funds to spend (because the address has never been used)'))
        puts(colored.cyan('0: All (works offline) - regardless of whether they have funds to spend (super advanced users only)'))
        puts(colored.cyan('\nb: Go Back\n'))
    choice = choice_prompt(
            user_prompt=DEFAULT_PROMPT,
            acceptable_responses=[0, 1, 2, 3],
            default_input='1',
            show_default=True,
            quit_ok=True,
            )

    if choice is False:
        return

    if choice == '1':
        return dump_selected_keys_or_addrs(wallet_obj=wallet_obj, zero_balance=False, used=True)
    elif choice == '2':
        return dump_selected_keys_or_addrs(wallet_obj=wallet_obj, zero_balance=True, used=True)
    elif choice == '3':
        return dump_selected_keys_or_addrs(wallet_obj=wallet_obj, zero_balance=None, used=False)
    elif choice == '0':
        return dump_all_keys_or_addrs(wallet_obj=wallet_obj)