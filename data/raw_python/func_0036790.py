def bump(db, aid):
    """Bump anime regular episode count."""
    anime = lookup(db, aid)
    if anime.complete:
        return
    episode = anime.watched_episodes + 1
    with db:
        set_watched(db, aid, get_eptype(db, 'regular').id, episode)
        set_status(
            db, aid,
            anime.enddate and episode >= anime.episodecount,
            episode)