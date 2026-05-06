def get_priority_rules(db) -> Iterable[PriorityRule]:
    """Get file priority rules."""
    cur = db.cursor()
    cur.execute('SELECT id, regexp, priority FROM file_priority')
    for row in cur:
        yield PriorityRule(*row)