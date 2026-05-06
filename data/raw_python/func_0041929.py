def result_to_dict(raw_result):
    """
    Parse raw result from fetcher into readable dictionary

    Args:
        raw_result (dict) - raw data from `fetcher`

    Returns:
        dict - readable dictionary
    """

    result = {}

    for channel_index, channel in enumerate(raw_result):
        channel_id, channel_name = channel[0], channel[1]
        channel_result = {
            'id': channel_id,
            'name': channel_name,
            'movies': []
        }
        for movie in channel[2]:
            channel_result['movies'].append({
                'title': movie[1],
                'start_time': datetime.fromtimestamp(movie[2]),
                'end_time': datetime.fromtimestamp(movie[2] + movie[3]),
                'inf': True if movie[3] else False,
            })
        result[channel_id] = channel_result

    return result