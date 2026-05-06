def set_watched(db, aid, ep_type, number):
    """Set episode as watched."""
    db.cursor().execute(
        """UPDATE episode SET user_watched=:watched
        WHERE aid=:aid AND type=:type AND number=:number""",
        {
            'aid': aid,
            'type': ep_type,
            'number': number,
            'watched': 1,
        })