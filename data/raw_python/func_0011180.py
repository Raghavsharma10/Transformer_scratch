def dump():
    """Print current environment

    Environment is outputted in a YAML-friendly format

    \b
    Usage:
        $ be dump
        Prefixed:
        - BE_TOPICS=hulk bruce animation
        - ...

    """

    if not self.isactive():
        lib.echo("ERROR: Enter a project first")
        sys.exit(lib.USER_ERROR)

    # Print custom environment variables first
    custom = sorted(os.environ.get("BE_ENVIRONMENT", "").split())
    if custom:
        lib.echo("Custom:")
        for key in custom:
            lib.echo("- %s=%s" % (key, os.environ.get(key)))

    # Then print redirected variables
    project = os.environ["BE_PROJECT"]
    root = os.environ["BE_PROJECTSROOT"]
    be = _extern.load(project, "be", optional=True, root=root)
    redirect = be.get("redirect", {}).items()
    if redirect:
        lib.echo("\nRedirect:")
        for map_source, map_dest in sorted(redirect):
            lib.echo("- %s=%s" % (map_dest, os.environ.get(map_dest)))

    # And then everything else
    prefixed = dict((k, v) for k, v in os.environ.iteritems()
                    if k.startswith("BE_"))
    if prefixed:
        lib.echo("\nPrefixed:")
        for key in sorted(prefixed):
            if not key.startswith("BE_"):
                continue
            lib.echo("- %s=%s" % (key, os.environ.get(key)))

    sys.exit(lib.NORMAL)