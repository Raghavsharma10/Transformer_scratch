def dashboard(request):
    """Shows the latest results for each source"""
    sources = (models.Source.objects.all().prefetch_related('metric_set')
                                          .order_by('name'))
    metrics = SortedDict([(src, src.metric_set.all()) for src in sources])
    no_source_metrics = models.Metric.objects.filter(source__isnull=True)
    if no_source_metrics:
        metrics[''] = no_source_metrics

    if request.META.get('HTTP_X_PJAX', False):
        parent_template = 'pjax.html'
    else:
        parent_template = 'base.html'
    return render(request, 'metrics/dashboard.html', {
        'source_metrics': metrics,
        'parent_template': parent_template
    })