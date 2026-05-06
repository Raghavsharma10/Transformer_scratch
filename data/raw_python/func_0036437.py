def rollback():
    """
    Rolls back to previous release
    """

    init_tasks()

    run_hook("before_rollback")

    # Remove current version
    current_release = paths.get_current_release_path()
    if current_release:
        env.run("rm -rf %s" % current_release)

    # Restore previous version
    old_release = paths.get_current_release_name()
    if old_release:
        paths.symlink(paths.get_source_path(old_release),
                      paths.get_current_path())

    run_hook("rollback")
    run_hook("after_rollback")

    logger.info("Rollback complete")