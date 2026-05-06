def recalculate_estimate(recalculate_total=False):
    """ Recalculate price of consumables that were used by resource until now.

        Regular task. It is too expensive to calculate consumed price on each
        request, so we store cached price each hour.
        If recalculate_total is True - task also recalculates total estimate
        for current month.
    """
    # Celery does not import server.urls and does not discover cost tracking modules.
    # So they should be discovered implicitly.
    CostTrackingRegister.autodiscover()
    # Step 1. Recalculate resources estimates.
    for resource_model in CostTrackingRegister.registered_resources:
        for resource in resource_model.objects.all():
            _update_resource_consumed(resource, recalculate_total=recalculate_total)
    # Step 2. Move from down to top and recalculate consumed estimate for each
    #         object based on its children.
    ancestors_models = [m for m in models.PriceEstimate.get_estimated_models()
                        if not issubclass(m, structure_models.ResourceMixin)]
    for model in ancestors_models:
        for ancestor in model.objects.all():
            _update_ancestor_consumed(ancestor)