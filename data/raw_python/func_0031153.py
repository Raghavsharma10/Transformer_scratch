def service(request, slug):
    "Displays a list of all services and their current status."
    
    try:
        service = Service.objects.get(slug=slug)
    except Service.DoesNotExist:
        return HttpResponseRedirect(reverse('overseer:index'))
        
    event_list = service.event_set.order_by('-date_created')
    
    return respond('overseer/service.html', {
        'service': service,
        'event_list': event_list,
    }, request)