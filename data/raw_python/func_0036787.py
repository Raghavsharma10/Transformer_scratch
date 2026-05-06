def add_episode(db, aid, episode):
    """Add an episode."""
    values = {
        'aid': aid,
        'type': episode.type,
        'number': episode.number,
        'title': episode.title,
        'length': episode.length,
    }
    upsert(db, 'episode', ['aid', 'type', 'number'], values)