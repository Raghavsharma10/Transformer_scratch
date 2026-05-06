def update(access_key, resource_policy, is_admin, is_active,  rate_limit):
    '''
    Update an existing keypair.

    ACCESS_KEY: Access key of an existing key pair.
    '''
    with Session() as session:
        try:
            data = session.KeyPair.update(
                access_key,
                is_active=is_active,
                is_admin=is_admin,
                resource_policy=resource_policy,
                rate_limit=rate_limit)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if not data['ok']:
            print_fail('KeyPair creation has failed: {0}'.format(data['msg']))
            sys.exit(1)
        print('Key pair is updated: ' + access_key + '.')