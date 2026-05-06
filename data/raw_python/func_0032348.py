def _getPublicSignupInfo(siteStore):
    """
    Get information about public web-based signup mechanisms.

    @param siteStore: a store with some signups installed on it (as indicated
    by _SignupTracker instances).

    @return: a generator which yields 2-tuples of (prompt, url) where 'prompt'
    is unicode briefly describing the signup mechanism (e.g. "Sign Up"), and
    'url' is a (unicode) local URL linking to a page where an anonymous user
    can access it.
    """

    # Note the underscore; this _should_ be a public API but it is currently an
    # unfortunate hack; there should be a different powerup interface that
    # requires prompt and prefixURL attributes rather than _SignupTracker.
    # -glyph

    for tr in siteStore.query(_SignupTracker):
        si = tr.signupItem
        p = getattr(si, 'prompt', None)
        u = getattr(si, 'prefixURL', None)
        if p is not None and u is not None:
            yield (p, u'/'+u)