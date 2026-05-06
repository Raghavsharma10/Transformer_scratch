def new(preset, name, silent, update):
    """Create new default preset

    \b
    Usage:
        $ be new ad
        "blue_unicorn" created
        $ be new film --name spiderman
        "spiderman" created

    """

    if self.isactive():
        lib.echo("Please exit current preset before starting a new")
        sys.exit(lib.USER_ERROR)

    if not name:
        count = 0
        name = lib.random_name()
        while name in _extern.projects():
            if count > 10:
                lib.echo("ERROR: Couldn't come up with a unique name :(")
                sys.exit(lib.USER_ERROR)
            name = lib.random_name()
            count += 1

    project_dir = lib.project_dir(_extern.cwd(), name)
    if os.path.exists(project_dir):
        lib.echo("\"%s\" already exists" % name)
        sys.exit(lib.USER_ERROR)

    username, preset = ([None] + preset.split("/", 1))[-2:]
    presets_dir = _extern.presets_dir()
    preset_dir = os.path.join(presets_dir, preset)

    # Is the preset given referring to a repository directly?
    relative = False if username else True

    try:
        if not update and preset in _extern.local_presets():
            _extern.copy_preset(preset_dir, project_dir)

        else:
            lib.echo("Finding preset for \"%s\".. " % preset, silent)
            time.sleep(1 if silent else 0)

            if relative:
                # Preset is relative, look it up from the Hub
                presets = _extern.github_presets()

                if preset not in presets:
                    sys.stdout.write("\"%s\" not found" % preset)
                    sys.exit(lib.USER_ERROR)

                time.sleep(1 if silent else 0)
                repository = presets[preset]

            else:
                # Absolute paths are pulled directly
                repository = username + "/" + preset

            lib.echo("Pulling %s.. " % repository, silent)
            repository = _extern.fetch_release(repository)

            # Remove existing preset
            if preset in _extern.local_presets():
                _extern.remove_preset(preset)

            try:
                _extern.pull_preset(repository, preset_dir)
            except IOError as e:
                lib.echo("ERROR: Sorry, something went wrong.\n"
                         "Use be --verbose for more")
                lib.echo(e)
                sys.exit(lib.USER_ERROR)

            try:
                _extern.copy_preset(preset_dir, project_dir)
            finally:
                # Absolute paths are not stored locally
                if not relative:
                    _extern.remove_preset(preset)

    except IOError as exc:
        if self.verbose:
            lib.echo("ERROR: %s" % exc)
        else:
            lib.echo("ERROR: Could not write, do you have permission?")
        sys.exit(lib.PROGRAM_ERROR)

    lib.echo("\"%s\" created" % name, silent)