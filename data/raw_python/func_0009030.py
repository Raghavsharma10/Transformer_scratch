def create_url(artist, song):
    """Create the URL in the LyricWikia format"""
    return (__BASE_URL__ +
            '/wiki/{artist}:{song}'.format(artist=urlize(artist),
                                           song=urlize(song)))