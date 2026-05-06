def set_status(
        db,
        aid: int,
        complete: Any,
        watched_episodes: int,
) -> None:
    """Set anime status."""
    upsert(db, 'cache_anime', ['aid'], {
        'aid': aid,
        'complete': 1 if complete else 0,
        'watched_episodes': watched_episodes,
    })