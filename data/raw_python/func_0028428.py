def delete(name):
    '''Delete the given virtual folder. This operation is irreversible!

    NAME: Name of a virtual folder.
    '''
    with Session() as session:
        try:
            session.VFolder(name).delete()
            print_done('Deleted.')
        except Exception as e:
            print_error(e)
            sys.exit(1)