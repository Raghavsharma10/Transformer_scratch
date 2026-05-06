def update(preset, clean):
    """Update a local preset

    This command will cause `be` to pull a preset already
    available locally.

    \b
    Usage:
        $ be update ad
        Updating "ad"..

    """

    if self.isactive():
        lib.echo("ERROR: Exit current project first")
        sys.exit(lib.USER_ERROR)

    presets = _extern.github_presets()

    if preset not in presets:
        sys.stdout.write("\"%s\" not found" % preset)
        sys.exit(lib.USER_ERROR)

    lib.echo("Are you sure you want to update \"%s\", "
             "any changes will be lost?: [y/N]: ", newline=False)
    if raw_input().lower() in ("y", "yes"):
        presets_dir = _extern.presets_dir()
        preset_dir = os.path.join(presets_dir, preset)

        repository = presets[preset]

        if clean:
            try:
                _extern.remove_preset(preset)
            except:
                lib.echo("ERROR: Could not clean existing preset")
                sys.exit(lib.USER_ERROR)

        lib.echo("Updating %s.. " % repository)

        try:
            _extern.pull_preset(repository, preset_dir)
        except IOError as e:
            lib.echo(e)
            sys.exit(lib.USER_ERROR)

    else:
        lib.echo("Cancelled")