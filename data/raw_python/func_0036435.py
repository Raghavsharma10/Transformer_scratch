def setup():
    """
    Creates shared and upload directory then fires setup to recipes.
    """

    init_tasks()

    run_hook("before_setup")

    # Create shared folder
    env.run("mkdir -p %s" % (paths.get_shared_path()))
    env.run("chmod 755 %s" % (paths.get_shared_path()))

    # Create backup folder
    env.run("mkdir -p %s" % (paths.get_backup_path()))
    env.run("chmod 750 %s" % (paths.get_backup_path()))

    # Create uploads folder
    env.run("mkdir -p %s" % (paths.get_upload_path()))
    env.run("chmod 775 %s" % (paths.get_upload_path()))

    run_hook("setup")
    run_hook("after_setup")