def command(state, args):
    """Register watching regexp for an anime."""
    args = parser.parse_args(args[1:])
    aid = state.results.parse_aid(args.aid, default_key='db')
    if args.query:
        # Use regexp provided by user.
        regexp = '.*'.join(args.query)
    else:
        # Make default regexp.
        title = query.select.lookup(state.db, aid, fields=['title']).title
        # Replace non-word, non-whitespace with whitespace.
        regexp = re.sub(r'[^\w\s]', ' ', title)
        # Split on whitespace and join with wildcard regexp.
        regexp = '.*?'.join(re.escape(x) for x in regexp.split())
        # Append episode matching regexp.
        regexp = '.*?'.join((
            regexp,
            r'\b(?P<ep>[0-9]+)(v[0-9]+)?',
        ))
    query.files.set_regexp(state.db, aid, regexp)