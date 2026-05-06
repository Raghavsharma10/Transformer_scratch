def preset_ls(remote):
    """List presets

    \b
    Usage:
        $ be preset ls
        - ad
        - game
        - film

    """

    if self.isactive():
        lib.echo("ERROR: Exit current project first")
        sys.exit(lib.USER_ERROR)

    if remote:
        presets = _extern.github_presets()
    else:
        presets = _extern.local_presets()

    if not presets:
        lib.echo("No presets found")
        sys.exit(lib.NORMAL)
    for preset in sorted(presets):
        lib.echo("- %s" % preset)
    sys.exit(lib.NORMAL)