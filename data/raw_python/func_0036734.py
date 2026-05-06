def get_episode_types(db) -> Iterator[EpisodeType]:
    """Get all episode types."""
    cur = db.cursor()
    cur.execute('SELECT id, name, prefix FROM episode_type')
    for type_id, name, prefix in cur:
        yield EpisodeType(type_id, name, prefix)