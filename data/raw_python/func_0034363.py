def clean_video_data(_data):
    """
    Clean video data:
        -> cleans title
        -> ...

    Args:
        _data (dict): Information about the video.

    Returns:
        dict: Refined video data.

    """

    data = _data.copy()

    # TODO: fix this ugliness
    title = data.get('title')
    if title:
        data['title'] = clean_title(title)

    return data