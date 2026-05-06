def freeze(wait, force_kill):
    '''Freeze manager.'''
    if wait and force_kill:
        print('You cannot use both --wait and --force-kill options '
              'at the same time.', file=sys.stderr)
        return

    with Session() as session:
        if wait:
            while True:
                resp = session.Manager.status()
                active_sessions_num = resp['active_sessions']
                if active_sessions_num == 0:
                    break
                print_wait('Waiting for all sessions terminated... ({0} left)'
                           .format(active_sessions_num))
                time.sleep(3)
            print_done('All sessions are terminated.')

        if force_kill:
            print_wait('Killing all sessions...')

        session.Manager.freeze(force_kill=force_kill)

        if force_kill:
            print_done('All sessions are killed.')

        print('Manager is successfully frozen.')