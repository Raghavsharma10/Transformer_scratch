def migrate(app_name=None, revision=None):
    "Syncs and migrates the database using South."
    cmd = ['python bin/manage.py syncdb --migrate']
    if app_name:
        cmd.append(app_name)
        if revision:
            cmd.append(revision)
    verun(' '.join(cmd))