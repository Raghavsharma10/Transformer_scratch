def get_lyrics(artist, song, linesep='\n', timeout=None):
    """Retrieve the lyrics of the song and return the first one in case
    multiple versions are available."""
    return get_all_lyrics(artist, song, linesep, timeout)[0]