def backup_db(release=None, limit=5):
    """
    Backup database and associate it with current release
    """

    assert "psql_user" in env, "Missing psql_user in env"
    assert "psql_db" in env, "Missing psql_db in env"
    assert "psql_password" in env, "Missing psql_password in env"

    if not release:
        release = paths.get_current_release_name()

    max_versions = limit+1

    if not release:
        logger.info("No releases present, skipping task")
        return

    remote_file = "postgresql/%s.sql.tar.gz" % release
    remote_path = paths.get_backup_path(remote_file)

    env.run("mkdir -p %s" % paths.get_backup_path("postgresql"))

    with context_managers.shell_env(PGPASSWORD=env.psql_password):
        env.run("pg_dump -h localhost -Fc -f %s -U %s %s -x -O" % (
            remote_path, env.psql_user, env.psql_db
        ))

    # Remove older releases
    env.run("ls -dt %s/* | tail -n +%s | xargs rm -rf" % (
        paths.get_backup_path("postgresql"),
        max_versions)
    )