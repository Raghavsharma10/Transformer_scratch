def is_enabled():
    """Returns ``True`` if bcrypt should be used."""
    enabled = getattr(settings, "BCRYPT_ENABLED", True)
    if not enabled:
        return False
    # Are we under a test?
    if hasattr(mail, 'outbox'):
        return getattr(settings, "BCRYPT_ENABLED_UNDER_TEST", False)
    return True