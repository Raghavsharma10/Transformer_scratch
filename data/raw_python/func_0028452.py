def terminate(sess_id_or_alias, owner, stats):
    '''
    Terminate the given session.

    SESSID: session ID or its alias given when creating the session.
    '''
    print_wait('Terminating the session(s)...')
    with Session() as session:
        has_failure = False
        for sess in sess_id_or_alias:
            try:
                kernel = session.Kernel(sess, owner)
                ret = kernel.destroy()
            except BackendAPIError as e:
                print_error(e)
                if e.status == 404:
                    print_info(
                        'If you are an admin, use "-o" / "--owner" option '
                        'to terminate other user\'s session.')
                has_failure = True
            except Exception as e:
                print_error(e)
                has_failure = True
            if has_failure:
                sys.exit(1)
        else:
            print_done('Done.')
            if stats:
                stats = ret.get('stats', None) if ret else None
                if stats:
                    print(_format_stats(stats))
                else:
                    print('Statistics is not available.')