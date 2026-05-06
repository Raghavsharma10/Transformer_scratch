def resource_quota_update(sender, instance, **kwargs):
    """ Update resource consumption details and price estimate if its configuration has changed """
    quota = instance
    resource = quota.scope
    try:
        new_configuration = CostTrackingRegister.get_configuration(resource)
    except ResourceNotRegisteredError:
        return
    models.PriceEstimate.update_resource_estimate(
        resource, new_configuration, raise_exception=not _is_in_celery_task())