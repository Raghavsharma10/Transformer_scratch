def command(state, args):
    """Fix cache issues caused by schema pre-v4."""
    if len(args) > 1:
        print(f'Usage: {args[0]}')
        return
    db = state.db
    _refresh_incomplete_anime(db)
    _fix_cached_completed(db)