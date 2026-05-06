def resource_update(sender, instance, created=False, **kwargs):
    """ Update resource consumption details and price estimate if its configuration has changed.
        Create estimates for previous months if resource was created not in current month.
    """
    resource = instance
    try:
        new_configuration = CostTrackingRegister.get_configuration(resource)
    except ResourceNotRegisteredError:
        return
    models.PriceEstimate.update_resource_estimate(
        resource, new_configuration, raise_exception=not _is_in_celery_task())
    # Try to create historical price estimates
    if created:
        _create_historical_estimates(resource, new_configuration)