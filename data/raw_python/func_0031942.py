def record_run(record_type, print_session_id, **kwds):
    """
    Record shell history.
    """
    if print_session_id and record_type != 'init':
        raise RuntimeError(
            '--print-session-id should be used with --record-type=init')

    cfstore = ConfigStore()
    # SOMEDAY: Pass a list of environment variables to shell by "rash
    # init" and don't read configuration in "rash record" command.  It
    # is faster.
    config = cfstore.get_config()
    envkeys = config.record.environ[record_type]
    json_path = os.path.join(cfstore.record_path,
                             record_type,
                             time.strftime('%Y-%m-%d-%H%M%S.json'))
    mkdirp(os.path.dirname(json_path))

    # Command line options directly map to record keys
    data = dict((k, v) for (k, v) in kwds.items() if v is not None)
    data.update(
        environ=get_environ(envkeys),
    )

    # Automatically set some missing variables:
    data.setdefault('cwd', getcwd())
    if record_type in ['command', 'exit']:
        data.setdefault('stop', int(time.time()))
    elif record_type in ['init']:
        data.setdefault('start', int(time.time()))

    if print_session_id:
        data['session_id'] = generate_session_id(data)
        print(data['session_id'])

    with open(json_path, 'w') as fp:
        json.dump(data, fp)