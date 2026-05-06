def require_invoke_minversion(min_version, verbose=False):
    """Ensures that :mod:`invoke` has at the least the :param:`min_version`.
    Otherwise,

    :param min_version: Minimal acceptable invoke version (as string).
    :param verbose:     Indicates if invoke.version should be shown.
    :raises: VersionRequirementError=SystemExit if requirement fails.
    """
    # -- REQUIRES: sys.path is setup and contains invoke
    try:
        import invoke
        invoke_version = invoke.__version__
    except ImportError:
        invoke_version = "__NOT_INSTALLED"

    if invoke_version < min_version:
        message = "REQUIRE: invoke.version >= %s (but was: %s)" % \
                  (min_version, invoke_version)
        message += "\nUSE: pip install invoke>=%s" % min_version
        raise VersionRequirementError(message)

    INVOKE_VERSION = os.environ.get("INVOKE_VERSION", None)
    if verbose and not INVOKE_VERSION:
        os.environ["INVOKE_VERSION"] = invoke_version
        print("USING: invoke.version=%s" % invoke_version)