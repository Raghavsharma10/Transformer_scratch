def command(state, args):
    """Reset anime watched episodes."""
    args = parser.parse_args(args[1:])
    aid = state.results.parse_aid(args.aid, default_key='db')
    query.update.reset(state.db, aid, args.episode)