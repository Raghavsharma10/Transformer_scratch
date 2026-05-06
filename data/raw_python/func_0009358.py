def blacken(c):
    """
    Install and execute ``black`` under appropriate circumstances, with diffs.

    Installs and runs ``black`` under Python 3.6 (the first version it
    supports). Since this sort of CI based task only needs to run once per
    commit (formatting is not going to change between interpreters) this seems
    like a worthwhile tradeoff.

    This task uses black's ``--check`` and ``--fail`` flags, so not only will
    the build fail if it does not conform, but contributors can see exactly
    what they need to change. This is intended as a hedge against the fact that
    not all contributors will be using Python 3.6+.
    """
    if not PYTHON.startswith("3.6"):
        msg = "Not blackening, since Python {} != Python 3.6".format(PYTHON)
        print(msg, file=sys.stderr)
        return
    # Install, allowing config override of hardcoded default version
    config = c.config.get("travis", {}).get("black", {})
    version = config.get("version", "18.5b0")
    c.run("pip install black=={}".format(version))
    # Execute our blacken task, with diff + check, which will both error
    # and emit diffs.
    checks.blacken(c, check=True, diff=True)