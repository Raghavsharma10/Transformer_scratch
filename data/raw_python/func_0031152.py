def index(request):
    "Displays a list of all services and their current status."
    
    service_list = Service.objects.all()
    
    event_list = list(Event.objects\
                             .filter(Q(status__gt=0) | Q(date_updated__gte=datetime.datetime.now()-datetime.timedelta(days=1)))\
                             .order_by('-date_created')[0:6])
    
    if event_list:
        latest_event, event_list = event_list[0], event_list[1:]
    else:
        latest_event = None
    
    return respond('overseer/index.html', {
        'service_list': service_list,
        'event_list': event_list,
        'latest_event': latest_event,
    }, request)