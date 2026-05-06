def _log_request(request):
    """Helper function to dump out debug info."""
    logger.debug("Inbound email received")

    for k, v in list(request.POST.items()):
        logger.debug("- POST['%s']='%s'" % (k, v))

    for n, f in list(request.FILES.items()):
        logger.debug("- FILES['%s']: '%s', %sB", n, f.content_type, f.size)