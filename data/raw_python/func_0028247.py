def delete(name):
    """
    Delete a keypair resource policy.

    NAME: NAME of a keypair resource policy to delete.
    """
    with Session() as session:
        if input('Are you sure? (y/n): ').lower().strip()[:1] != 'y':
            print('Canceled.')
            sys.exit(1)
        try:
            data = session.ResourcePolicy.delete(name)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        if not data['ok']:
            print_fail('KeyPair Resource Policy deletion has failed: {0}'
                       .format(data['msg']))
            sys.exit(1)
        print('Resource policy ' + name + ' is deleted.')