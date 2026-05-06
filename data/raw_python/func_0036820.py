def cache_files(db, aid: int, anime_files: AnimeFiles) -> None:
    """Cache files for anime."""
    with db:
        cache_status(db, aid)
        db.cursor().execute(
            """UPDATE cache_anime
            SET anime_files=?
            WHERE aid=?""",
            (anime_files.to_json(), aid))