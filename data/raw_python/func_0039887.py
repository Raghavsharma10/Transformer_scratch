def initiate_session(config):
    """
    Initiate a session globally used in prof :
      + Retrive the cookie
      + Log to prof

    Returns an initiated session
    """
    global baseurl
    baseurl = config['DEFAULT']['baseurl']
    if 'session' in config['DEFAULT']:
        cookies = {
            'PHPSESSID': config['DEFAULT']['session']
        }
        prof_session.cookies = requests.utils.cookiejar_from_dict(cookies)
    try:
        valid = verify_session(prof_session, baseurl)
        if not valid:
            # Looks like this session is not valid anymore, try to get a new one
            get_session(prof_session, baseurl, config)
        return prof_session
    except:
        print("{baseurl} not reachable. Verify your connection".format(baseurl=baseurl))
        exit(1)