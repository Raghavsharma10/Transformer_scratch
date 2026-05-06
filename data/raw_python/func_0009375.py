def build(c, sdist=True, wheel=False, directory=None, python=None, clean=True):
    """
    Build sdist and/or wheel archives, optionally in a temp base directory.

    All parameters save ``directory`` honor config settings of the same name,
    under the ``packaging`` tree. E.g. say ``.configure({'packaging': {'wheel':
    True}})`` to force building wheel archives by default.

    :param bool sdist:
        Whether to build sdists/tgzs.

    :param bool wheel:
        Whether to build wheels (requires the ``wheel`` package from PyPI).

    :param str directory:
        Allows specifying a specific directory in which to perform builds and
        dist creation. Useful when running as a subroutine from ``publish``
        which sets up a temporary directory.

        Two subdirectories will be created within this directory: one for
        builds, and one for the dist archives.

        When ``None`` or another false-y value, the current working directory
        is used (and thus, local ``dist/`` and ``build/`` subdirectories).

    :param str python:
        Which Python binary to use when invoking ``setup.py``.

        Defaults to just ``python``.

        If ``wheel=True``, then this Python must have ``wheel`` installed in
        its default ``site-packages`` (or similar) location.

    :param bool clean:
        Whether to clean out the local ``build/`` folder before building.
    """
    # Config hooks
    config = c.config.get("packaging", {})
    # TODO: update defaults to be None, then flip the below so non-None runtime
    # beats config.
    sdist = config.get("sdist", sdist)
    wheel = config.get("wheel", wheel)
    python = config.get("python", python or "python")  # buffalo buffalo
    # Sanity
    if not sdist and not wheel:
        sys.exit(
            "You said no sdists and no wheels..."
            "what DO you want to build exactly?"
        )
    # Directory path/arg logic
    if not directory:
        directory = ""  # os.path.join() doesn't like None
    dist_dir = os.path.join(directory, "dist")
    dist_arg = "-d {0}".format(dist_dir)
    build_dir = os.path.join(directory, "build")
    build_arg = "-b {0}".format(build_dir)
    # Clean
    if clean:
        if os.path.exists(build_dir):
            rmtree(build_dir)
        # NOTE: not cleaning dist_dir, since this may be called >1 time within
        # publish() trying to build up multiple wheels/etc.
        # TODO: separate clean-build/clean-dist args? Meh
    # Build
    parts = [python, "setup.py"]
    if sdist:
        parts.extend(("sdist", dist_arg))
    if wheel:
        # Manually execute build in case we are using a custom build dir.
        # Doesn't seem to be a way to tell bdist_wheel to do this directly.
        parts.extend(("build", build_arg))
        parts.extend(("bdist_wheel", dist_arg))
    c.run(" ".join(parts))