def list_hosts():
    '''List the hosts of virtual folders that is accessible to the current user.'''
    with Session() as session:
        try:
            resp = session.VFolder.list_hosts()
            print(f"Default vfolder host: {resp['default']}")
            print(f"Usable hosts: {', '.join(resp['allowed'])}")
        except Exception as e:
            print_error(e)
            sys.exit(1)