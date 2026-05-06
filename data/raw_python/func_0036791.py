def reset(db, aid, episode):
    """Reset episode count for anime."""
    params = {
        'aid': aid,
        'type': get_eptype(db, 'regular').id,
        'watched': 1,
        'number': episode,
    }
    with db:
        cur = db.cursor()
        cur.execute(
            """UPDATE episode SET user_watched=:watched
            WHERE aid=:aid AND type=:type AND number<=:number""",
            params)
        params['watched'] = 0
        cur.execute(
            """UPDATE episode SET user_watched=:watched
            WHERE aid=:aid AND type=:type AND number>:number""",
            params)
        cache_status(db, aid, force=True)