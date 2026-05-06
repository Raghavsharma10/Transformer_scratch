def info(name):
    '''Show the information of the given virtual folder.

    NAME: Name of a virtual folder.
    '''
    with Session() as session:
        try:
            result = session.VFolder(name).info()
            print('Virtual folder "{0}" (ID: {1})'
                  .format(result['name'], result['id']))
            print('- Owner:', result['is_owner'])
            print('- Permission:', result['permission'])
            print('- Number of files: {0}'.format(result['numFiles']))
        except Exception as e:
            print_error(e)
            sys.exit(1)