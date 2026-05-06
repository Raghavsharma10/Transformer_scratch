def api_send_mail(request, key=None, hproPk=None):
    """Send a email. Posts parameters are used"""

    if not check_api_key(request, key, hproPk):
        return HttpResponseForbidden

    sender = request.POST.get('sender', settings.MAIL_SENDER)
    dests = request.POST.getlist('dests')
    subject = request.POST['subject']
    message = request.POST['message']
    html_message = request.POST.get('html_message')

    if html_message and html_message.lower() == 'false':
        html_message = False

    if 'response_id' in request.POST:
        key = hproPk + ':' + request.POST['response_id']
    else:
        key = None

    generic_send_mail(sender, dests, subject, message, key, 'PlugIt API (%s)' % (hproPk or 'StandAlone',), html_message)

    return HttpResponse(json.dumps({}), content_type="application/json")