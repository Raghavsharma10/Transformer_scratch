def _validation(response):
    """
    Get the validation value for a challenge response.
    """
    h = hashlib.sha256(response.key_authorization.encode("utf-8"))
    return b64encode(h.digest()).decode()