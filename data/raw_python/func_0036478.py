def command(state, args):
    """Search Animanager database."""

    args = parser.parse_args(args[1:])
    where_queries = []
    params = {}
    if args.watching or args.available:
        where_queries.append('regexp IS NOT NULL')
    if args.query:
        where_queries.append('title LIKE :title')
        params['title'] = _compile_sql_query(args.query)
    if not where_queries:
        print('Must include at least one filter.')
        return
    where_query = ' AND '.join(where_queries)
    logger.debug('Search where %s with params %s', where_query, params)

    results = list()
    all_files = [
        filename for filename in _find_files(state.config['anime'].getpath('watchdir'))
        if _is_video(filename)
    ]
    for anime in query.select.select(state.db, where_query, params):
        logger.debug('For anime %s with regexp %s', anime.aid, anime.regexp)
        if anime.regexp is not None:
            anime_files = AnimeFiles(anime.regexp, all_files)
            logger.debug('Found files %s', anime_files.filenames)
            query.files.cache_files(state.db, anime.aid, anime_files)
            available = anime_files.available_string(anime.watched_episodes)
        else:
            available = ''
        if not args.available or available:
            results.append((
                anime.aid, anime.title, anime.type,
                '{}/{}'.format(anime.watched_episodes, anime.episodecount),
                'yes' if anime.complete else '',
                available,
            ))
    state.results['db'].set(results)
    state.results['db'].print()