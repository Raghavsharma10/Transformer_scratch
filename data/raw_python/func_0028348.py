def heartbeat(request):
    """
    Runs all the Django checks and returns a JsonResponse with either
    a status code of 200 or 500 depending on the results of the checks.

    Any check that returns a warning or worse (error, critical) will
    return a 500 response.
    """
    all_checks = checks.registry.registry.get_checks(
        include_deployment_checks=not settings.DEBUG,
    )

    details = {}
    statuses = {}
    level = 0

    for check in all_checks:
        detail = heartbeat_check_detail(check)
        statuses[check.__name__] = detail['status']
        level = max(level, detail['level'])
        if detail['level'] > 0:
            details[check.__name__] = detail

    if level < checks.messages.WARNING:
        status_code = 200
        heartbeat_passed.send(sender=heartbeat, level=level)
    else:
        status_code = 500
        heartbeat_failed.send(sender=heartbeat, level=level)

    payload = {
        'status': level_to_text(level),
        'checks': statuses,
        'details': details,
    }
    return JsonResponse(payload, status=status_code)