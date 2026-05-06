def deploy(app_id, version, promote, quiet):
    # type: (str, str, bool, bool) -> None
    """ Deploy the app to AppEngine.

    Args:
        app_id (str):
            AppEngine App ID. Overrides config value app_id if given.
        version (str):
            AppEngine project version. Overrides config values if given.
        promote (bool):
            If set to **True** promote the current remote app version to the one
            that's being deployed.
        quiet (bool):
            If set to **True** this will pass the ``--quiet`` flag to gcloud
            command.
    """
    gae_app = GaeApp.for_branch(git.current_branch().name)

    if gae_app is None and None in (app_id,  version):
        msg = (
            "Can't find an AppEngine app setup for branch <35>{}<32> and"
            "--project and --version were not given."
        )
        log.err(msg, git.current_branch().name)
        sys.exit(1)

    if version is not None:
        gae_app.version = version

    if app_id is not None:
        gae_app.app_id = app_id

    gae_app.deploy(promote, quiet)