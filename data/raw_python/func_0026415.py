def update(ctx, no_restart, no_rebuild):
    """Update a HFOS node"""

    # 0. (NOT YET! MAKE A BACKUP OF EVERYTHING)
    # 1. update repository
    # 2. update frontend repository
    # 3. (Not yet: update venv)
    # 4. rebuild frontend
    # 5. restart service

    instance = ctx.obj['instance']

    log('Pulling github updates')
    run_process('.', ['git', 'pull', 'origin', 'master'])
    run_process('./frontend', ['git', 'pull', 'origin', 'master'])

    if not no_rebuild:
        log('Rebuilding frontend')
        install_frontend(instance, forcerebuild=True, install=False, development=True)

    if not no_restart:
        log('Restaring service')
        if instance != 'hfos':
            instance = 'hfos-' + instance

        run_process('.', ['sudo', 'systemctl', 'restart', instance])

    log('Done')