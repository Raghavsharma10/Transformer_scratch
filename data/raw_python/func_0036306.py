def restore_db(release=None):
    """
    Restores backup back to version, uses current version by default.
    """

    assert "mysql_user" in env, "Missing mysqL_user in env"
    assert "mysql_password" in env, "Missing mysql_password in env"
    assert "mysql_host" in env, "Missing mysql_host in env"
    assert "mysql_db" in env, "Missing mysql_db in env"

    if not release:
        release = paths.get_current_release_name()

    if not release:
        raise Exception("Release %s was not found" % release)

    backup_file = "mysql/%s.sql.gz" % release
    backup_path = paths.get_backup_path(backup_file)

    if not env.exists(backup_path):
        raise Exception("Backup file %s not found" % backup_path)

    env.run("gunzip < %s | mysql -u %s -p%s -h %s %s" %
            (backup_path, env.mysql_user, env.mysql_password, env.mysql_host,
             env.mysql_db))