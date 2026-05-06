def request_anime(aid: int) -> 'Anime':
    """Make an anime API request."""
    anime_info = alib.request_anime(_CLIENT, aid)
    return Anime._make(anime_info)