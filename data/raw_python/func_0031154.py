def event(request, id):
    "Displays a list of all services and their current status."
    
    try:
        evt = Event.objects.get(pk=id)
    except Event.DoesNotExist:
        return HttpResponseRedirect(reverse('overseer:index'))
    
    update_list = list(evt.eventupdate_set.order_by('-date_created'))
    
    return respond('overseer/event.html', {
        'event': evt,
        'update_list': update_list,
    }, request)