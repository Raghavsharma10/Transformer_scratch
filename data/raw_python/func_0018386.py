def peep_install(argv):
    """Perform the ``peep install`` subcommand, returning a shell status code
    or raising a PipException.

    :arg argv: The commandline args, starting after the subcommand

    """
    output = []
    out = output.append
    reqs = []
    try:
        req_paths = list(requirement_args(argv, want_paths=True))
        if not req_paths:
            out("You have to specify one or more requirements files with the -r option, because\n"
                "otherwise there's nowhere for peep to look up the hashes.\n")
            return COMMAND_LINE_ERROR

        # We're a "peep install" command, and we have some requirement paths.
        reqs = list(chain.from_iterable(
            downloaded_reqs_from_path(path, argv)
            for path in req_paths))
        buckets = bucket(reqs, lambda r: r.__class__)

        # Skip a line after pip's "Cleaning up..." so the important stuff
        # stands out:
        if any(buckets[b] for b in ERROR_CLASSES):
            out('\n')

        printers = (lambda r: out(r.head()),
                    lambda r: out(r.error() + '\n'),
                    lambda r: out(r.foot()))
        for c in ERROR_CLASSES:
            first_every_last(buckets[c], *printers)

        if any(buckets[b] for b in ERROR_CLASSES):
            out('-------------------------------\n'
                'Not proceeding to installation.\n')
            return SOMETHING_WENT_WRONG
        else:
            for req in buckets[InstallableReq]:
                req.install()

            first_every_last(buckets[SatisfiedReq], *printers)

        return ITS_FINE_ITS_FINE
    except (UnsupportedRequirementError, InstallationError, DownloadError) as exc:
        out(str(exc))
        return SOMETHING_WENT_WRONG
    finally:
        for req in reqs:
            req.dispose()
        print(''.join(output))