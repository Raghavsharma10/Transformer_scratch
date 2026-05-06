def ls(name, path):
    """
    List files in a path of a virtual folder.

    \b
    NAME: Name of a virtual folder.
    PATH: Path inside vfolder.
    """
    with Session() as session:
        try:
            print_wait('Retrieving list of files in "{}"...'.format(path))
            result = session.VFolder(name).list_files(path)
            if 'error_msg' in result and result['error_msg']:
                print_fail(result['error_msg'])
                return
            files = json.loads(result['files'])
            table = []
            headers = ['file name', 'size', 'modified', 'mode']
            for file in files:
                mdt = datetime.fromtimestamp(file['mtime'])
                mtime = mdt.strftime('%b %d %Y %H:%M:%S')
                row = [file['filename'], file['size'], mtime, file['mode']]
                table.append(row)
            print_done('Retrived.')
            print(tabulate(table, headers=headers))
        except Exception as e:
            print_error(e)