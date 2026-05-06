def unindex_layers_with_issues(self, use_cache=False):
    """
    Remove the index for layers in search backend, which are linked to an issue.
    """
    from hypermap.aggregator.models import Issue, Layer, Service
    from django.contrib.contenttypes.models import ContentType

    layer_type = ContentType.objects.get_for_model(Layer)
    service_type = ContentType.objects.get_for_model(Service)

    for issue in Issue.objects.filter(content_type__pk=layer_type.id):
        unindex_layer(issue.content_object.id, use_cache)

    for issue in Issue.objects.filter(content_type__pk=service_type.id):
        for layer in issue.content_object.layer_set.all():
            unindex_layer(layer.id, use_cache)