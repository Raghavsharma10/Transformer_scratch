def command(state, args):
    """Search AniDB."""
    args = parser.parse_args(args[1:])
    if not args.query:
        print('Must supply query.')
        return
    search_query = _compile_re_query(args.query)
    results = state.titles.search(search_query)
    results = [(anime.aid, anime.main_title) for anime in results]
    state.results['anidb'].set(results)
    state.results['anidb'].print()