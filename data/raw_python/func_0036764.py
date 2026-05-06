def command(state, args):
    """Add an anime from an AniDB search."""
    if len(args) < 2:
        print(f'Usage: {args[0]} {{ID|aid:AID}}')
        return
    aid = state.results.parse_aid(args[1], default_key='anidb')
    anime = request_anime(aid)
    query.update.add(state.db, anime)