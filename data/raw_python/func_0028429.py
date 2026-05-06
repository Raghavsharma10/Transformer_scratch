def rename(old_name, new_name):
    '''Rename the given virtual folder. This operation is irreversible!
    You cannot change the vfolders that are shared by other users,
    and the new name must be unique among all your accessible vfolders
    including the shared ones.

    OLD_NAME: The current name of a virtual folder.
    NEW_NAME: The new name of a virtual folder.
    '''
    with Session() as session:
        try:
            session.VFolder(old_name).rename(new_name)
            print_done('Renamed.')
        except Exception as e:
            print_error(e)
            sys.exit(1)