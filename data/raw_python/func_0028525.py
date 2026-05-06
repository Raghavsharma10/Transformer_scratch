def ls(sess_id_or_alias, path):
    """
    List files in a path of a running container.

    \b
    SESSID: Session ID or its alias given when creating the session.
    PATH: Path inside container.
    """
    with Session() as session:
        try:
            print_wait('Retrieving list of files in "{}"...'.format(path))
            kernel = session.Kernel(sess_id_or_alias)
            result = kernel.list_files(path)

            if 'errors' in result and result['errors']:
                print_fail(result['errors'])
                sys.exit(1)

            files = json.loads(result['files'])
            table = []
            headers = ['file name', 'size', 'modified', 'mode']
            for file in files:
                mdt = datetime.fromtimestamp(file['mtime'])
                mtime = mdt.strftime('%b %d %Y %H:%M:%S')
                row = [file['filename'], file['size'], mtime, file['mode']]
                table.append(row)
            print_done('Retrived.')
            print('Path in container:', result['abspath'], end='')
            print(tabulate(table, headers=headers))
        except Exception as e:
            print_error(e)
            sys.exit(1)