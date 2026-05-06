def clean(exclude):
    # type: (bool, List[str]) -> None
    """ Remove all unnecessary files.

    Args:
        pretend (bool):
            If set to **True**, do not delete any files, just show what would be
            deleted.
        exclude (list[str]):
            A list of path patterns to exclude from deletion.
    """
    pretend = context.get('pretend', False)
    exclude = list(exclude) + conf.get('clean.exclude', [])
    clean_patterns = conf.get('clean.patterns', [
        '*__pycache__*',
        '*.py[cod]',
        '*.swp',
    ])

    num_files = 0
    with util.timed_block() as t:
        files = fs.filtered_walk(conf.proj_path(), clean_patterns, exclude)
        for path in files:
            try:
                num_files += 1

                if not isdir(path):
                    log.info('  <91>[file] <90>{}', path)
                    not pretend and os.remove(path)
                else:
                    log.info('  <91>[dir]  <90>{}', path)
                    not pretend and rmtree(path)

            except OSError:
                log.info("<33>Failed to remove <90>{}", path)

    if pretend:
        msg = "Would delete <33>{}<32> files. Took <33>{}<32>s"
    else:
        msg = "Deleted <33>{}<32> files in <33>{}<32>s"

    log.info(msg.format(num_files, t.elapsed_s))