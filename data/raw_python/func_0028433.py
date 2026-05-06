def rm(name, filenames, recursive):
    '''
    Delete files in a virtual folder.
    If one of the given paths is a directory and the recursive option is enabled,
    all its content and the directory itself are recursively deleted.

    This operation is irreversible!

    \b
    NAME: Name of a virtual folder.
    FILENAMES: Paths of the files to delete.
    '''
    with Session() as session:
        try:
            if input("> Are you sure? (y/n): ").lower().strip()[:1] == 'y':
                session.VFolder(name).delete_files(
                    filenames,
                    recursive=recursive)
                print_done('Done.')
        except Exception as e:
            print_error(e)
            sys.exit(1)