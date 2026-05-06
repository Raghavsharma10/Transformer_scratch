def add_handler(args):
    """usage: {program} add <source-file> <offset> <width> <context-width>

    Add a new anchor for a file.
    """
    file_path = pathlib.Path(args['<source-file>']).resolve()

    try:
        offset = int(args['<offset>'])
        width = int(args['<width>'])
        context_width = int(args['<context-width>'])
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return ExitCode.DATAERR

    repo = _open_repo(args, '<source-file>')

    if sys.stdin.isatty():
        text = _launch_editor('# json metadata')
    else:
        text = sys.stdin.read()

    try:
        metadata = json.loads(text)
    except json.JSONDecodeError:
        print(
            'Failed to create anchor. Invalid JSON metadata.', file=sys.stderr)
        return ExitCode.DATAERR

    # TODO: let user specify encoding
    with file_path.open(mode='rt') as handle:
        anchor = make_anchor(
            file_path, offset, width, context_width, metadata, handle=handle)

    repo.add(anchor)

    return ExitCode.OK