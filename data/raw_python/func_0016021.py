def vocab_account_type(instance):
    """Ensure a user-account objects' 'account-type' property is from the
    account-type-ov vocabulary.
    """
    for key, obj in instance['objects'].items():
        if 'type' in obj and obj['type'] == 'user-account':
            try:
                acct_type = obj['account_type']
            except KeyError:
                continue
            if acct_type not in enums.ACCOUNT_TYPE_OV:
                yield JSONError("Object '%s' is a User Account Object "
                                "with an 'account_type' of '%s', which is not a "
                                "value in the account-type-ov vocabulary."
                                % (key, acct_type), instance['id'], 'account-type')