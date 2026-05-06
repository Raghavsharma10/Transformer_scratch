def filter_files_extensions(files, extension_lists):
    """
    Put the files in buckets according to extension_lists
    files=[movie.avi, movie.srt], extension_lists=[[avi],[srt]] ==> [[movie.avi],[movie.srt]]
    :param files: A list of files
    :param extension_lists: A list of list of extensions
    :return: The files filtered and sorted according to extension_lists
    """
    log.debug('filter_files_extensions: files="{}"'.format(files))
    result = [[] for _ in extension_lists]
    for file in files:
        ext = file.suffix[1:].lower()
        for ext_i, ext_list in enumerate(extension_lists):
            if ext in ext_list:
                result[ext_i].append(file)
    log.debug('filter_files_extensions result:{}'.format(result))
    return result