def sync_local_to_remote(force="no"):
    """
    Sync your local postgres database with remote

    Example:
        fabrik prod sync_local_to_remote:force=yes
    """

    _check_requirements()

    if force != "yes":
        message = "This will replace the remote database '%s' with your "\
            "local '%s', are you sure [y/n]" % (env.psql_db, env.local_psql_db)
        answer = prompt(message, "y")

        if answer != "y":
            logger.info("Sync stopped")
            return

    init_tasks()  # Bootstrap fabrik

    # Create database dump
    local_file = "sync_%s.sql.tar.gz" % int(time.time()*1000)
    local_path = "/tmp/%s" % local_file

    with context_managers.shell_env(PGPASSWORD=env.local_psql_password):
        elocal("pg_dump -h localhost -Fc -f %s -U %s %s -x -O" % (
            local_path, env.local_psql_user, env.local_psql_db
        ))

    remote_path = "/tmp/%s" % local_file

    # Upload sync file
    put(remote_path, local_path)

    # Import sync file by performing the following task (drop, create, import)
    with context_managers.shell_env(PGPASSWORD=env.psql_password):
        env.run("pg_restore --clean -h localhost -d %s -U %s '%s'" % (
            env.psql_db,
            env.psql_user,
            remote_path)
        )

    # Cleanup
    env.run("rm %s" % remote_path)
    elocal("rm %s" % local_path)

    # Trigger hook
    run_hook("postgres.after_sync_local_to_remote")

    logger.info("Sync complete")