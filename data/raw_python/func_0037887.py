def cleanup_callbackredirect(max_age=300):
    """
    Delete old :model:`esi.CallbackRedirect` models.
    Accepts a max_age parameter, in seconds (default 300).
    """
    max_age = timezone.now() - timedelta(seconds=max_age)
    logger.debug("Deleting all callback redirects created before {0}".format(max_age.strftime("%b %d %Y %H:%M:%S")))
    CallbackRedirect.objects.filter(created__lte=max_age).delete()