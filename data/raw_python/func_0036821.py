def get_files(conn, aid: int) -> AnimeFiles:
    """Get cached files for anime."""
    with conn:
        cur = conn.cursor().execute(
            'SELECT anime_files FROM cache_anime WHERE aid=?',
            (aid,))
        row = cur.fetchone()
        if row is None:
            raise ValueError('No cached files')
        return AnimeFiles.from_json(row[0])