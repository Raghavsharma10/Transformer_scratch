def get_complete(db) -> Iterator[int]:
    """Return AID of complete anime."""
    cur = db.cursor()
    cur.execute(
        """SELECT aid FROM cache_anime
        WHERE complete=?""", (1,))
    for row in cur:
        yield row[0]