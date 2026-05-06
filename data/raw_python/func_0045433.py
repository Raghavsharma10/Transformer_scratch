def _get_video_ts_file_paths(dvd_path):
    """Returns a sorted list of paths for files contained in th VIDEO_TS folder of the specified
       DVD path.
    """

    video_ts_folder_path = join(dvd_path, "VIDEO_TS")

    video_ts_file_paths = []

    for video_ts_folder_content_name in listdir(video_ts_folder_path):
        video_ts_folder_content_path = join(video_ts_folder_path, video_ts_folder_content_name)

        if isfile(video_ts_folder_content_path):
            video_ts_file_paths.append(video_ts_folder_content_path)

    return sorted(video_ts_file_paths)