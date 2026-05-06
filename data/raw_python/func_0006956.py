def rsync():
    """ syncs the src code to the remote box """
    log_green('syncing code to remote box...')
    data = load_state_from_disk()
    if 'SOURCE_PATH' in os.environ:
        with lcd(os.environ['SOURCE_PATH']):
            local("rsync  -a "
                  "--info=progress2 "
                  "--exclude .git "
                  "--exclude .tox "
                  "--exclude .vagrant "
                  "--exclude venv "
                  ". "
                  "-e 'ssh -C -i " + env.ec2_key_filename + "' "
                  "%s@%s:" % (env.user, data['ip_address']))
    else:
        print('please export SOURCE_PATH before running rsync')
        exit(1)