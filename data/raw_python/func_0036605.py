def cache_status(db, aid, force=False):
    """Calculate and cache status for given anime.

    Don't do anything if status already exists and force is False.

    """
    with db:
        cur = db.cursor()
        if not force:
            # We don't do anything if we already have this aid in our
            # cache.
            cur.execute('SELECT 1 FROM cache_anime WHERE aid=?', (aid,))
            if cur.fetchone() is not None:
                return

        # Retrieve information for determining complete.
        cur.execute(
            'SELECT episodecount, enddate FROM anime WHERE aid=?', (aid,))
        row = cur.fetchone()
        if row is None:
            raise ValueError('aid provided does not exist')
        episodecount, enddate = row

        # Select all regular episodes in ascending order.
        cur.execute("""
            SELECT number, user_watched FROM episode
            WHERE aid=? AND type=?
            ORDER BY number ASC
            """, (aid, get_eptype(db, 'regular').id))

        # We find the last consecutive episode that is user_watched.
        number = 0
        for number, watched in cur:
            # Once we find the first unwatched episode, we set the last
            # consecutive watched episode to the previous episode (or 0).
            if watched == 0:
                number -= 1
                break
        # We store this in the cache.
        set_status(db, aid, enddate and episodecount <= number, number)