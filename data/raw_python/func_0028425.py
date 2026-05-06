def list():
    '''List virtual folders that belongs to the current user.'''
    fields = [
        ('Name', 'name'),
        ('ID', 'id'),
        ('Owner', 'is_owner'),
        ('Permission', 'permission'),
    ]
    with Session() as session:
        try:
            resp = session.VFolder.list()
            if not resp:
                print('There is no virtual folders created yet.')
                return
            rows = (tuple(vf[key] for _, key in fields) for vf in resp)
            hdrs = (display_name for display_name, _ in fields)
            print(tabulate(rows, hdrs))
        except Exception as e:
            print_error(e)
            sys.exit(1)