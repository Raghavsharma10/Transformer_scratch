def details_handler(args):
    """usage: {program} details <anchor-id> [<path>]

    Get the details of a single anchor.
    """

    repo = _open_repo(args)
    _, anchor = _get_anchor(repo, args['<anchor-id>'])

    print("""path: {file_path}
encoding: {encoding}

[before]
{before}
--------------

[topic]
{topic}
--------------

[after]
{after}
--------------

offset: {offset}
width: {width}""".format(
        file_path=anchor.file_path,
        encoding=anchor.encoding,
        before=anchor.context.before,
        topic=anchor.context.topic,
        after=anchor.context.after,
        offset=anchor.context.offset,
        width=anchor.context.width))

    return ExitCode.OK