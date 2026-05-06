def delete_episode(db, aid, episode):
    """Delete an episode."""
    db.cursor().execute(
        'DELETE FROM episode WHERE aid=:aid AND type=:type AND number=:number',
        {
            'aid': aid,
            'type': episode.type,
            'number': episode.number,
        })