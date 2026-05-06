def delete(access_key):
    """
    Delete an existing keypair.

    ACCESSKEY: ACCESSKEY for a keypair to delete.
    """
    with Session() as session:
        try:
            data = session.KeyPair.delete(access_key)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if not data['ok']:
            print_fail('KeyPair deletion has failed: {0}'.format(data['msg']))
            sys.exit(1)
        print('Key pair is deleted: ' + access_key + '.')