def merge_path_subvideo(path_subvideos, callback):
    """
    Merge subtitles into videos.
    :param path_subvideos: a dict with paths as key and a list of lists of videos and subtitles
    :param callback: Instance of ProgressCallback
    :return: tuple with list of videos and list of subtitles (videos have matched subtitles)
    """
    log.debug('merge_path_subvideo(path_subvideos=<#paths={nb_paths}>)'.format(nb_paths=len(path_subvideos)))
    # FIXME: add logging
    nb_videos = sum([len(subvids[1]) for subvids in path_subvideos.values()])

    all_videos = []
    all_subtitles = []

    callback.set_range(0, nb_videos)

    vid_i = 0
    callback.update(vid_i)
    for path, subvideos in path_subvideos.items():
        [subs_path, vids_path] = subvideos
        subtitles = [LocalSubtitleFile(filepath=sub_path) for sub_path in subs_path]
        all_subtitles.extend(subtitles)
        for vid_path in vids_path:
            try:
                video = VideoFile(vid_path)
            except NotAVideoException:
                continue
            all_videos.append(video)

            for subtitle in subtitles:
                if subtitle.matches_video_filename(video):
                    video.add_subtitle(subtitle)
            video.get_subtitles().add_candidates(subtitles)

            vid_i += 1
            callback.update(vid_i)
    callback.finish(True)
    return all_videos, all_subtitles