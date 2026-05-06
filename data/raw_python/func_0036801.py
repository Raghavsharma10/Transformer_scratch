def command(state, args):
    """Add a priority rule for files."""
    args = parser.parse_args(args[1:])
    row_id = query.files.add_priority_rule(state.db, args.regexp, args.priority)
    del state.file_picker
    print('Added rule {}'.format(row_id))