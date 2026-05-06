def mkdir(name, path):
    '''Create an empty directory in the virtual folder.

    \b
    NAME: Name of a virtual folder.
    PATH: The name or path of directory. Parent directories are created automatically
          if they do not exist.
    '''
    with Session() as session:
        try:
            session.VFolder(name).mkdir(path)
            print_done('Done.')
        except Exception as e:
            print_error(e)
            sys.exit(1)