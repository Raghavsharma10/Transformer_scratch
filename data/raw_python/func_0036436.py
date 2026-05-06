def deploy():
    """
    Performs a deploy by invoking copy, then generating next release name and
    invoking necessary hooks.
    """

    init_tasks()

    if not has_hook("copy"):
        return report("No copy method has been defined")

    if not env.exists(paths.get_shared_path()):
        return report("You need to run setup before running deploy")

    run_hook("before_deploy")

    release_name = int(time.time()*1000)
    release_path = paths.get_releases_path(release_name)

    env.current_release = release_path

    try:
        run_hook("copy")
    except Exception as e:
        return report("Error occurred on copy. Aborting deploy", err=e)

    if not env.exists(paths.get_source_path(release_name)):
        return report("Source path not found '%s'" %
                      paths.get_source_path(release_name))

    try:
        run_hook("deploy")
    except Exception as e:
        message = "Error occurred on deploy, starting rollback..."

        logger.error(message)
        logger.error(e)

        run_task("rollback")
        return report("Error occurred on deploy")

    # Symlink current folder
    paths.symlink(paths.get_source_path(release_name),
                  paths.get_current_path())

    # Clean older releases
    if "max_releases" in env:
        cleanup_releases(int(env.max_releases))

    run_hook("after_deploy")

    if "public_path" in env:
        paths.symlink(paths.get_source_path(release_name), env.public_path)

    logger.info("Deploy complete")