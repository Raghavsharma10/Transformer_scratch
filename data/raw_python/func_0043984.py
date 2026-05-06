def list_handler(args):
    """usage: {program} list

    List the anchors for a file.
    """
    repo = open_repository(None)
    for anchor_id, anchor in repo.items():
        print("{} {}:{} => {}".format(anchor_id,
                                      anchor.file_path.relative_to(repo.root),
                                      anchor.context.offset, anchor.metadata))

    return ExitCode.OK