def scan_videopath(videopath, callback, recursive=False):
    """
    Scan the videopath string for video files.
    :param videopath: Path object
    :param callback: Instance of ProgressCallback
    :param recursive: True if the scanning should happen recursive
    :return: tuple with list of videos and list of subtitles (videos have matched subtitles)
    """
    log.debug('scan_videopath(videopath="{videopath}", recursive={recursive})'.format(
        videopath=videopath, recursive=recursive))
    if not videopath.exists():
        log.debug('"{videopath}" does not exist'.format(videopath=videopath))
        raise IllegalPathException(path=videopath)
    if videopath.is_dir():
        log.debug('"{videopath}" is a directory'.format(videopath=videopath))
        return __scan_folder(videopath, callback=callback, recursive=recursive)
    elif videopath.is_file():
        log.debug('"{videopath}" is a file'.format(videopath=videopath))
        videopath_dir = videopath.parent
        [all_subs, _] = filter_files_extensions(videopath_dir.iterdir(), [SUBTITLES_EXT, VIDEOS_EXT])
        [_, video] = filter_files_extensions([videopath], [SUBTITLES_EXT, VIDEOS_EXT])
        sub_videos = [all_subs, video]
        path_subvideos = {videopath_dir: sub_videos}
        return merge_path_subvideo(path_subvideos, callback)
    else:
        log.debug('"{videopath}" is of unknown type'.format(videopath=videopath))
        return [], []