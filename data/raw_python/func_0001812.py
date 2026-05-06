def credentials_checker(url, username, password):
    """Check the provided credentials using the Harvest API."""
    api = HarvestAPI(url, (username, password))
    try:
        api.whoami()
    except HarvestError:
        return False
    else:
        return True