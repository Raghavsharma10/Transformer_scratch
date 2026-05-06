def last_event(request, slug):
    "Displays a list of all services and their current status."
    
    try:
        service = Service.objects.get(slug=slug)
    except Service.DoesNotExist:
        return HttpResponseRedirect(reverse('overseer:index'))
    
    try:
        evt = service.event_set.order_by('-date_created')[0]
    except IndexError:
        return HttpResponseRedirect(service.get_absolute_url())
    
    return event(request, evt.pk)