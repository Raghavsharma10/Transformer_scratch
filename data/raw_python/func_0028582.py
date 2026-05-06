def keypair():
    '''
    Show the server-side information of the currently configured access key.
    '''
    fields = [
        ('User ID', 'user_id'),
        ('Access Key', 'access_key'),
        ('Secret Key', 'secret_key'),
        ('Active?', 'is_active'),
        ('Admin?', 'is_admin'),
        ('Created At', 'created_at'),
        ('Last Used', 'last_used'),
        ('Res.Policy', 'resource_policy'),
        ('Rate Limit', 'rate_limit'),
        ('Concur.Limit', 'concurrency_limit'),
        ('Concur.Used', 'concurrency_used'),
    ]
    with Session() as session:
        try:
            kp = session.KeyPair(session.config.access_key)
            info = kp.info(fields=(item[1] for item in fields))
        except Exception as e:
            print_error(e)
            sys.exit(1)
        rows = []
        for name, key in fields:
            rows.append((name, info[key]))
        print(tabulate(rows, headers=('Field', 'Value')))