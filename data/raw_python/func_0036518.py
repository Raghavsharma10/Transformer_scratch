def command(state, args):
    """Add an anime from an AniDB search."""
    args = parser.parse_args(args[1:])
    if args.watching:
        rows = query.select.select(state.db, 'regexp IS NOT NULL', [], ['aid'])
        aids = [anime.aid for anime in rows]
    elif args.incomplete:
        rows = query.select.select(state.db, 'enddate IS NULL', [], ['aid'])
        aids = [anime.aid for anime in rows]
    else:
        aid = state.results.parse_aid(args.aid, default_key='db')
        aids = [aid]
    if not aids:
        return
    anime = request_anime(aids.pop())
    query.update.add(state.db, anime)
    print('Updated {} {}'.format(anime.aid, anime.title))
    for aid in aids:
        time.sleep(2)
        anime = request_anime(aid)
        query.update.add(state.db, anime)
        print('Updated {} {}'.format(anime.aid, anime.title))