def download(name, filenames):
    '''
    Download a file from the virtual folder to the current working directory.
    The files with the same names will be overwirtten.

    \b
    NAME: Name of a virtual folder.
    FILENAMES: Paths of the files to be uploaded.
    '''
    with Session() as session:
        try:
            session.VFolder(name).download(filenames, show_progress=True)
            print_done('Done.')
        except Exception as e:
            print_error(e)
            sys.exit(1)