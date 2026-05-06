def create(name, host):
    '''Create a new virtual folder.

    \b
    NAME: Name of a virtual folder.
    HOST: Name of a virtual folder host in which the virtual folder will be created.
    '''
    with Session() as session:
        try:
            result = session.VFolder.create(name, host)
            print('Virtual folder "{0}" is created.'.format(result['name']))
        except Exception as e:
            print_error(e)
            sys.exit(1)