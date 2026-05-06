def command(state, args):
    """Show anime data."""
    args = parser.parse_args(args[1:])
    aid = state.results.parse_aid(args.aid, default_key='db')
    anime = query.select.lookup(state.db, aid, episode_fields=args.episode_fields)

    complete_string = 'yes' if anime.complete else 'no'
    print(SHOW_MSG.format(
        anime.aid,
        anime.title,
        anime.type,
        anime.watched_episodes,
        anime.episodecount,
        datets.to_date(anime.startdate) if anime.startdate else 'N/A',
        datets.to_date(anime.enddate) if anime.enddate else 'N/A',
        complete_string,
    ))
    if anime.regexp:
        print('Watching regexp: {}'.format(anime.regexp))
    if hasattr(anime, 'episodes'):
        episodes = sorted(anime.episodes, key=lambda x: (x.type, x.number))
        print('\n', tabulate(
            (
                (
                    EpisodeTypes.from_db(state.db).get_epno(episode),
                    episode.title,
                    episode.length,
                    'yes' if episode.user_watched else '',
                )
                for episode in episodes
            ),
            headers=['Number', 'Title', 'min', 'Watched'],
        ))