def return_input_paths(job, work_dir, ids, *args):
    """
    Returns the paths of files from the FileStore

    Input1: Toil job instance
    Input2: Working directory
    Input3: jobstore id dictionary
    Input4: names of files to be returned from the jobstore

    Returns: path(s) to the file(s) requested -- unpack these!
    """
    paths = OrderedDict()
    for name in args:
        if not os.path.exists(os.path.join(work_dir, name)):
            file_path = job.fileStore.readGlobalFile(ids[name], os.path.join(work_dir, name))
        else:
            file_path = os.path.join(work_dir, name)
        paths[name] = file_path
        if len(args) == 1:
            return file_path

    return paths.values()