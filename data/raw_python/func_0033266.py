def get_template_filelist(repo_path, ignore_files=[], ignore_folders=[]):
    """
    input: local repo path
    output: path list of files which need to be rendered
    """

    default_ignore_files = ['.gitignore']
    default_ignore_folders = ['.git']

    ignore_files += default_ignore_files
    ignore_folders += default_ignore_folders

    filelist = []

    for root, folders, files in os.walk(repo_path):
        for ignore_file in ignore_files:
            if ignore_file in files:
                files.remove(ignore_file)

        for ignore_folder in ignore_folders:
            if ignore_folder in folders:
                folders.remove(ignore_folder)

        for file_name in files:
            filelist.append( '%s/%s' % (root, file_name))

    return filelist