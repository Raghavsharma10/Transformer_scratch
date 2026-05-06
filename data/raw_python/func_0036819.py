def delete_priority_rule(db, rule_id: int) -> None:
    """Delete a file priority rule."""
    with db:
        cur = db.cursor()
        cur.execute('DELETE FROM file_priority WHERE id=?', (rule_id,))