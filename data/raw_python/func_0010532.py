def __scan_folder(folder_path, callback, recursive=False):
    """
    Scan a folder for videos and subtitles
    :param folder_path: String of a directory
    :param callback: Instance of ProgressCallback
    :param recursive: True if the scanning should happen recursive
    :return: tuple with list of videos and list of subtitles (videos have matched subtitles)
    """
    log.debug('__scan_folder(folder_path="{folder_path}", recursive={recursive})'.format(folder_path=folder_path,
                                                                                         recursive=recursive))
    path_subvideos = {}
    # FIXME: a folder named 'movie.avi' is also considered a movie. Fix this.
    if recursive:
        for dir_path, _, files in os.walk(str(folder_path)):
            log.debug('walking current directory:"{}"'.format(dir_path))
            path_files = [Path(dir_path) / file for file in files]
            sub_videos = filter_files_extensions(path_files, [SUBTITLES_EXT, VIDEOS_EXT])
            path_subvideos[dir_path] = sub_videos
    else:
        files = [folder_path / f for f in folder_path.iterdir() if f.is_file()]  # filter(lambda f: (folder_path / f).is_file(), folder_path.iterdir())
        sub_videos = filter_files_extensions(files, [SUBTITLES_EXT, VIDEOS_EXT])
        path_subvideos[folder_path] = sub_videos
    return merge_path_subvideo(path_subvideos, callback)