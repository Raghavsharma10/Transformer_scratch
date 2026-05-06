def command(state, args):
    """Delete priority rule."""
    args = parser.parse_args(args[1:])
    query.files.delete_priority_rule(state.db, args.id)
    del state.file_picker