def webhooks_v2(request):
    """
    Handles all known webhooks from stripe, and calls signals.
    Plug in as you need.
    """
    if request.method != "POST":
        return HttpResponse("Invalid Request.", status=400)

    try:
        event_json = simplejson.loads(request.body)
    except AttributeError:
        # Backwords compatibility
        # Prior to Django 1.4, request.body was named request.raw_post_data
        event_json = simplejson.loads(request.raw_post_data)
    event_key = event_json['type'].replace('.', '_')

    if event_key in WEBHOOK_MAP:
        WEBHOOK_MAP[event_key].send(sender=None, full_json=event_json)

    return HttpResponse(status=200)