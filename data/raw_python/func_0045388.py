def get_profile(session):
    """Get complete profile."""
    try:
        profile = session.get(PROFILE_URL).json()
        if 'errorCode' in profile and profile['errorCode'] == '403':
            raise MoparError("not logged in")
        return profile
    except JSONDecodeError:
        raise MoparError("not logged in")