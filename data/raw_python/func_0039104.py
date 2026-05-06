def authenticate(org):
    """
    Authenticate with GitHub via SSH if possible
    Otherwise authenticate via HTTPS
    Returns an authenticated User
    """
    with ProgressBar(_("Authenticating")) as progress_bar:
        user = _authenticate_ssh(org)
        progress_bar.stop()
        if user is None:
            # SSH auth failed, fallback to HTTPS
            with _authenticate_https(org) as user:
                yield user
        else:
            yield user