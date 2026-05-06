def add_priority_rule(
        db, regexp: str, priority: Optional[int] = None,
) -> int:
    """Add a file priority rule."""
    with db:
        cur = db.cursor()
        if priority is None:
            cur.execute('SELECT MAX(priority) FROM file_priority')
            highest_priority = cur.fetchone()[0]
            if highest_priority is None:
                priority = 1
            else:
                priority = highest_priority + 1
        cur.execute("""
            INSERT INTO file_priority (regexp, priority)
            VALUES (?, ?)""", (regexp, priority))
        row_id = db.last_insert_rowid()
    return row_id