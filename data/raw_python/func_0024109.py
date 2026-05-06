def zapier_guest_hook(request):
    '''
    Zapier can POST something like this when tickets are bought:
    {

        "ticket_type": "Individual (Regular)",
        "barcode": "12345678",
        "email": "demo@example.com"
    }
    '''
    if request.META.get('HTTP_X_ZAPIER_SECRET', None) != settings.WAFER_TICKETS_SECRET:
        raise PermissionDenied('Incorrect secret')

    # This is required for python 3, and in theory fine on python 2
    payload = json.loads(request.body.decode('utf8'))
    import_ticket(payload['barcode'], payload['ticket_type'],
                  payload['email'])

    return HttpResponse("Noted\n", content_type='text/plain')