def prepare(c):
    """
    Edit changelog & version, git commit, and git tag, to set up for release.
    """
    # Print dry-run/status/actions-to-take data & grab programmatic result
    # TODO: maybe expand the enum-based stuff to have values that split up
    # textual description, command string, etc. See the TODO up by their
    # definition too, re: just making them non-enum classes period.
    # TODO: otherwise, we at least want derived eg changelog/version/etc paths
    # transmitted from status() into here...
    actions, state = status(c)
    # TODO: unless nothing-to-do in which case just say that & exit 0
    if not confirm("Take the above actions?"):
        sys.exit("Aborting.")

    # TODO: factor out what it means to edit a file:
    # - $EDITOR or explicit expansion of it in case no shell involved
    # - pty=True and hide=False, because otherwise things can be bad
    # - what else?

    # Changelog! (pty for non shite editing, eg vim sure won't like non-pty)
    if actions.changelog is Changelog.NEEDS_RELEASE:
        # TODO: identify top of list and inject a ready-made line? Requires vim
        # assumption...GREAT opportunity for class/method based tasks!
        cmd = "$EDITOR {0.packaging.changelog_file}".format(c)
        c.run(cmd, pty=True, hide=False)
    # TODO: add a step for checking reqs.txt / setup.py vs virtualenv contents
    # Version file!
    if actions.version == VersionFile.NEEDS_BUMP:
        # TODO: suggest the bump and/or overwrite the entire file? Assumes a
        # specific file format. Could be bad for users which expose __version__
        # but have other contents as well.
        version_file = os.path.join(
            _find_package(c),
            c.packaging.get("version_module", "_version") + ".py",
        )
        cmd = "$EDITOR {0}".format(version_file)
        c.run(cmd, pty=True, hide=False)
    if actions.tag == Tag.NEEDS_CUTTING:
        # Commit, if necessary, so the tag includes everything.
        # NOTE: this strips out untracked files. effort.
        cmd = 'git status --porcelain | egrep -v "^\\?"'
        if c.run(cmd, hide=True, warn=True).ok:
            c.run(
                'git commit -am "Cut {0}"'.format(state.expected_version),
                hide=False,
            )
        # Tag!
        c.run("git tag {0}".format(state.expected_version), hide=False)