def download(sess_id_or_alias, files, dest):
    """
    Download files from a running container.

    \b
    SESSID: Session ID or its alias given when creating the session.
    FILES: Paths inside container.
    """
    if len(files) < 1:
        return
    with Session() as session:
        try:
            print_wait('Downloading file(s) from {}...'
                       .format(sess_id_or_alias))
            kernel = session.Kernel(sess_id_or_alias)
            kernel.download(files, dest, show_progress=True)
            print_done('Downloaded to {}.'.format(dest.resolve()))
        except Exception as e:
            print_error(e)
            sys.exit(1)