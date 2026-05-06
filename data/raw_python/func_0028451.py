def start(lang, session_id, owner, env, mount, tag, resources, cluster_size):
    '''
    Prepare and start a single compute session without executing codes.
    You may use the created session to execute codes using the "run" command
    or connect to an application service provided by the session using the "app"
    command.


    \b
    LANG: The name (and version/platform tags appended after a colon) of session
          runtime or programming language.
    '''
    if session_id is None:
        session_id = token_hex(5)
    else:
        session_id = session_id

    ######
    envs = _prepare_env_arg(env)
    resources = _prepare_resource_arg(resources)
    mount = _prepare_mount_arg(mount)
    with Session() as session:
        try:
            kernel = session.Kernel.get_or_create(
                lang,
                client_token=session_id,
                cluster_size=cluster_size,
                mounts=mount,
                envs=envs,
                resources=resources,
                owner_access_key=owner,
                tag=tag)
        except Exception as e:
            print_error(e)
            sys.exit(1)
        else:
            if kernel.created:
                print_info('Session ID {0} is created and ready.'
                           .format(session_id))
            else:
                print_info('Session ID {0} is already running and ready.'
                           .format(session_id))
            if kernel.service_ports:
                print_info('This session provides the following app services: ' +
                           ', '.join(sport['name']
                                     for sport in kernel.service_ports))