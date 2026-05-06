def stc_system_info(stc_addr):
    """Return dictionary of STC and API information.

    If a session already exists, then use it to get STC information and avoid
    taking the time to start a new session.  A session is necessary to get
    STC information.

    """
    stc = stchttp.StcHttp(stc_addr)
    sessions = stc.sessions()
    if sessions:
        # If a session already exists, use it to get STC information.
        stc.join_session(sessions[0])
        sys_info = stc.system_info()
    else:
        # Create a new session to get STC information.
        stc.new_session('anonymous')
        try:
            sys_info = stc.system_info()
        finally:
            # Make sure the temporary session in terminated.
            stc.end_session()

    return sys_info