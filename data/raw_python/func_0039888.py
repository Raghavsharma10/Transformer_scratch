def verify_session(session, baseurl):
    """
    Check that this session is still valid on this baseurl, ie, we get a list of projects
    """
    request = session.post(baseurl+"/select_projet.php")
    return VERIFY_SESSION_STRING in request.content.decode('iso-8859-1')