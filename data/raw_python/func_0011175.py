def activate():
    """Enter into an environment with support for tab-completion

    This command drops you into a subshell, similar to the one
    generated via `be in ...`, except no topic is present and
    instead it enables tab-completion for supported shells.

    See documentation for further information.
    https://github.com/mottosso/be/wiki/cli

    """

    parent = lib.parent()

    try:
        cmd = lib.cmd(parent)
    except SystemError as exc:
        lib.echo(exc)
        sys.exit(lib.PROGRAM_ERROR)

    # Store reference to calling shell
    context = lib.context(root=_extern.cwd())
    context["BE_SHELL"] = parent

    if lib.platform() == "unix":
        context["BE_TABCOMPLETION"] = os.path.join(
            os.path.dirname(__file__), "_autocomplete.sh").replace("\\", "/")

    context.pop("BE_ACTIVE", None)

    sys.exit(subprocess.call(cmd, env=context))