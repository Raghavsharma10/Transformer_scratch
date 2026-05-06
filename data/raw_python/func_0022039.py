def domains(request):
    """
    A page with number of services and layers faceted on domains.
    """
    url = ''
    query = '*:*&facet=true&facet.limit=-1&facet.pivot=domain_name,service_id&wt=json&indent=true&rows=0'
    if settings.SEARCH_TYPE == 'elasticsearch':
        url = '%s/select?q=%s' % (settings.SEARCH_URL, query)
    if settings.SEARCH_TYPE == 'solr':
        url = '%s/solr/hypermap/select?q=%s' % (settings.SEARCH_URL, query)
    LOGGER.debug(url)
    response = urllib2.urlopen(url)
    data = response.read().replace('\n', '')
    # stats
    layers_count = Layer.objects.all().count()
    services_count = Service.objects.all().count()
    template = loader.get_template('aggregator/index.html')
    context = RequestContext(request, {
        'data': data,
        'layers_count': layers_count,
        'services_count': services_count,
    })
    return HttpResponse(template.render(context))