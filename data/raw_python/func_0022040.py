def tasks_runner(request):
    """
    A page that let the admin to run global tasks.
    """

    # server info
    cached_layers_number = 0
    cached_layers = cache.get('layers')
    if cached_layers:
        cached_layers_number = len(cached_layers)

    cached_deleted_layers_number = 0
    cached_deleted_layers = cache.get('deleted_layers')
    if cached_deleted_layers:
        cached_deleted_layers_number = len(cached_deleted_layers)

    # task actions
    if request.method == 'POST':
        if 'check_all' in request.POST:
            if settings.REGISTRY_SKIP_CELERY:
                check_all_services()
            else:
                check_all_services.delay()
        if 'index_all' in request.POST:
            if settings.REGISTRY_SKIP_CELERY:
                index_all_layers()
            else:
                index_all_layers.delay()
        if 'index_cached' in request.POST:
            if settings.REGISTRY_SKIP_CELERY:
                index_cached_layers()
            else:
                index_cached_layers.delay()
        if 'drop_cached' in request.POST:
            cache.set('layers', None)
            cache.set('deleted_layers', None)
        if 'clear_index' in request.POST:
            if settings.REGISTRY_SKIP_CELERY:
                clear_index()
            else:
                clear_index.delay()
        if 'remove_index' in request.POST:
            if settings.REGISTRY_SKIP_CELERY:
                unindex_layers_with_issues()
            else:
                unindex_layers_with_issues.delay()

    return render(
        request,
        'aggregator/tasks_runner.html', {
            'cached_layers_number': cached_layers_number,
            'cached_deleted_layers_number': cached_deleted_layers_number,
        }
    )