def upload(c, directory, index=None, sign=False, dry_run=False):
    """
    Upload (potentially also signing) all artifacts in ``directory``.

    :param str index:
        Custom upload index/repository name.

        By default, uses whatever the invoked ``pip`` is configured to use.
        Modify your ``pypirc`` file to add new named repositories.

    :param bool sign:
        Whether to sign the built archive(s) via GPG.

    :param bool dry_run:
        Skip actual publication step if ``True``.

        This also prevents cleanup of the temporary build/dist directories, so
        you can examine the build artifacts.
    """
    # Obtain list of archive filenames, then ensure any wheels come first
    # so their improved metadata is what PyPI sees initially (otherwise, it
    # only honors the sdist's lesser data).
    archives = list(
        itertools.chain.from_iterable(
            glob(os.path.join(directory, "dist", "*.{0}".format(extension)))
            for extension in ("whl", "tar.gz")
        )
    )
    # Sign each archive in turn
    # TODO: twine has a --sign option; but the below is still nice insofar
    # as it lets us dry-run, generate for web upload when pypi's API is
    # being cranky, etc. Figure out which is better.
    if sign:
        prompt = "Please enter GPG passphrase for signing: "
        input_ = StringIO(getpass.getpass(prompt) + "\n")
        gpg_bin = find_gpg(c)
        if not gpg_bin:
            sys.exit(
                "You need to have one of `gpg`, `gpg1` or `gpg2` "
                "installed to GPG-sign!"
            )
        for archive in archives:
            cmd = "{0} --detach-sign -a --passphrase-fd 0 {{0}}".format(
                gpg_bin
            )  # noqa
            c.run(cmd.format(archive), in_stream=input_)
            input_.seek(0)  # So it can be replayed by subsequent iterations
    # Upload
    parts = ["twine", "upload"]
    if index:
        index_arg = "--repository {0}".format(index)
    if index:
        parts.append(index_arg)
    paths = archives[:]
    if sign:
        paths.append(os.path.join(directory, "dist", "*.asc"))
    parts.extend(paths)
    cmd = " ".join(parts)
    if dry_run:
        print("Would publish via: {0}".format(cmd))
        print("Files that would be published:")
        c.run("ls -l {0}".format(" ".join(paths)))
    else:
        c.run(cmd)