def sync_remote_to_local(force="no"):
    """
    Sync your remote postgres database with local

    Example:
        fabrik prod sync_remote_to_local
    """

    _check_requirements()

    if force != "yes":
        message = "This will replace your local database '%s' with the "\
            "remote '%s', are you sure [y/n]" % (env.local_psql_db, env.psql_db)
        answer = prompt(message, "y")

        if answer != "y":
            logger.info("Sync stopped")
            return

    init_tasks()  # Bootstrap fabrik

    # Create database dump
    remote_file = "postgresql/sync_%s.sql.tar.gz" % int(time.time()*1000)
    remote_path = paths.get_backup_path(remote_file)

    env.run("mkdir -p %s" % paths.get_backup_path("postgresql"))

    with context_managers.shell_env(PGPASSWORD=env.psql_password):
        env.run("pg_dump -h localhost -Fc -f %s -U %s %s -x -O" % (
            remote_path, env.psql_user, env.psql_db
        ))

    local_path = "/tmp/%s" % remote_file

    # Download sync file
    get(remote_path, local_path)

    # Import sync file by performing the following task (drop, create, import)
    with context_managers.shell_env(PGPASSWORD=env.local_psql_password):
        elocal("pg_restore --clean -h localhost -d %s -U %s '%s'" % (
            env.local_psql_db,
            env.local_psql_user,
            local_path)
        )

    # Cleanup
    env.run("rm %s" % remote_path)
    elocal("rm %s" % local_path)

    # Trigger hook
    run_hook("postgres.after_sync_remote_to_local")

    logger.info("Sync complete")