def _get_user_agent(platform=None):
    """ Convenience function that looks up a user agent string, random if N/A
    """
    if isinstance(platform, ustr):
        platform = platform.upper()
    return {"chrome": AGENT_CHROME, "edge": AGENT_EDGE, "ios": AGENT_IOS}.get(
        platform, random.choice(AGENT_ALL)
    )