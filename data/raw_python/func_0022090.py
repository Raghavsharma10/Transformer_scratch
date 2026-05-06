def remove_service_checks(self, service_id):
    """
    Remove all checks from a service.
    """
    from hypermap.aggregator.models import Service
    service = Service.objects.get(id=service_id)

    service.check_set.all().delete()
    layer_to_process = service.layer_set.all()
    for layer in layer_to_process:
        layer.check_set.all().delete()