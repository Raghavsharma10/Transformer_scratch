def get_all_lyrics(artist, song, linesep='\n', timeout=None):
    """Retrieve a list of all the lyrics versions of a song."""
    url = create_url(artist, song)
    response = _requests.get(url, timeout=timeout)
    soup = _BeautifulSoup(response.content, "html.parser")
    lyricboxes = soup.findAll('div', {'class': 'lyricbox'})

    if not lyricboxes:
        raise LyricsNotFound('Cannot download lyrics')

    for lyricbox in lyricboxes:
        for br in lyricbox.findAll('br'):
            br.replace_with(linesep)

    return [lyricbox.text.strip() for lyricbox in lyricboxes]