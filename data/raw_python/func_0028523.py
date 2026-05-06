def upload(sess_id_or_alias, files):
    """
    Upload files to user's home folder.

    \b
    SESSID: Session ID or its alias given when creating the session.
    FILES: Path to upload.
    """
    if len(files) < 1:
        return
    with Session() as session:
        try:
            print_wait('Uploading files...')
            kernel = session.Kernel(sess_id_or_alias)
            kernel.upload(files, show_progress=True)
            print_done('Uploaded.')
        except Exception as e:
            print_error(e)
            sys.exit(1)