def render(request, template_name, context=None, content_type=None, status=None, using=None, logs=None):
    """
    Wrapper around Django render method. Can take one or a list of logs and logs the response.
    No overhead if no logs are passed.
    """
    if logs:
        obj_logger = ObjectLogger()
        if not isinstance(logs, list):
            logs = [logs, ]
        for log in logs:
            log = obj_logger.log_response(
                log,
                context,
                status=str(status),
                headers='',
                content_type=str(content_type))
            log.save()
    return django_render(
        request,
        template_name,
        context=context,
        content_type=content_type,
        status=status,
        using=using)