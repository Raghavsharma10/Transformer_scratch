def dry_run(func):
    """Dry run: simulate sql execution."""
    @wraps(func)
    def inner(dry_run, *args, **kwargs):
        ret = func(dry_run=dry_run, *args, **kwargs)
        if not dry_run:
            db.session.commit()
        else:
            db.session.rollback()
        return ret
    return inner