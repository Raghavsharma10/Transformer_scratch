def _resource_deletion(resource):
    """ Recalculate consumption details and save resource details """
    if resource.__class__ not in CostTrackingRegister.registered_resources:
        return
    new_configuration = {}
    price_estimate = models.PriceEstimate.update_resource_estimate(resource, new_configuration)
    price_estimate.init_details()