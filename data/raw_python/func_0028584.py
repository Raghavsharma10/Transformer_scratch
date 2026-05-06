def add(user_id, resource_policy, admin, inactive,  rate_limit):
    '''
    Add a new keypair.

    USER_ID: User ID of a new key pair.
    RESOURCE_POLICY: resource policy for new key pair.
    '''
    try:
        user_id = int(user_id)
    except ValueError:
        pass  # string-based user ID for Backend.AI v1.4+
    with Session() as session:
        try:
            data = session.KeyPair.create(
                user_id,
                is_active=not inactive,
                is_admin=admin,
                resource_policy=resource_policy,
                rate_limit=rate_limit)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if not data['ok']:
            print_fail('KeyPair creation has failed: {0}'.format(data['msg']))
            sys.exit(1)
        item = data['keypair']
        print('Access Key: {0}'.format(item['access_key']))
        print('Secret Key: {0}'.format(item['secret_key']))