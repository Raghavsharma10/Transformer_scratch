def command(state, args):
    """Unregister watching regexp for an anime."""
    args = parser.parse_args(args[1:])
    if args.complete:
        query.files.delete_regexp_complete(state.db)
    else:
        if args.aid is None:
            parser.print_help()
        else:
            aid = state.results.parse_aid(args.aid, default_key='db')
            query.files.delete_regexp(state.db, aid)