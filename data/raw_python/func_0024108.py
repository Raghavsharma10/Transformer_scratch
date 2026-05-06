def zapier_cancel_hook(request):
    '''
    Zapier can post something like this when tickets are cancelled
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
    ticket = Ticket.objects.filter(barcode=payload['barcode'])
    if ticket.exists():
        # delete the ticket
        ticket.delete()
    return HttpResponse("Cancelled\n", content_type='text/plain')