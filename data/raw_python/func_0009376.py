def publish(
    c,
    sdist=True,
    wheel=False,
    index=None,
    sign=False,
    dry_run=False,
    directory=None,
    dual_wheels=False,
    alt_python=None,
    check_desc=False,
):
    """
    Publish code to PyPI or index of choice.

    All parameters save ``dry_run`` and ``directory`` honor config settings of
    the same name, under the ``packaging`` tree. E.g. say
    ``.configure({'packaging': {'wheel': True}})`` to force building wheel
    archives by default.

    :param bool sdist:
        Whether to upload sdists/tgzs.

    :param bool wheel:
        Whether to upload wheels (requires the ``wheel`` package from PyPI).

    :param str index:
        Custom upload index/repository name. See ``upload`` help for details.

    :param bool sign:
        Whether to sign the built archive(s) via GPG.

    :param bool dry_run:
        Skip actual publication step if ``True``.

        This also prevents cleanup of the temporary build/dist directories, so
        you can examine the build artifacts.

    :param str directory:
        Base directory within which will live the ``dist/`` and ``build/``
        directories.

        Defaults to a temporary directory which is cleaned up after the run
        finishes.

    :param bool dual_wheels:
        When ``True``, builds individual wheels for Python 2 and Python 3.

        Useful for situations where you can't build universal wheels, but still
        want to distribute for both interpreter versions.

        Requires that you have a useful ``python3`` (or ``python2``, if you're
        on Python 3 already) binary in your ``$PATH``. Also requires that this
        other python have the ``wheel`` package installed in its
        ``site-packages``; usually this will mean the global site-packages for
        that interpreter.

        See also the ``alt_python`` argument.

    :param str alt_python:
        Path to the 'alternate' Python interpreter to use when
        ``dual_wheels=True``.

        When ``None`` (the default) will be ``python3`` or ``python2``,
        depending on the currently active interpreter.

    :param bool check_desc:
        Whether to run ``setup.py check -r -s`` (uses ``readme_renderer``)
        before trying to publish - catches long_description bugs. Default:
        ``False``.
    """
    # Don't hide by default, this step likes to be verbose most of the time.
    c.config.run.hide = False
    # Config hooks
    config = c.config.get("packaging", {})
    index = config.get("index", index)
    sign = config.get("sign", sign)
    dual_wheels = config.get("dual_wheels", dual_wheels)
    check_desc = config.get("check_desc", check_desc)
    # Initial sanity check, if needed. Will die usefully.
    if check_desc:
        c.run("python setup.py check -r -s")
    # Build, into controlled temp dir (avoids attempting to re-upload old
    # files)
    with tmpdir(skip_cleanup=dry_run, explicit=directory) as tmp:
        # Build default archives
        build(c, sdist=sdist, wheel=wheel, directory=tmp)
        # Build opposing interpreter archive, if necessary
        if dual_wheels:
            if not alt_python:
                alt_python = "python2"
                if sys.version_info[0] == 2:
                    alt_python = "python3"
            build(c, sdist=False, wheel=True, directory=tmp, python=alt_python)
        # Do the thing!
        upload(c, directory=tmp, index=index, sign=sign, dry_run=dry_run)