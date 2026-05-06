def scope_deletion(sender, instance, **kwargs):
    """ Run different actions on price estimate scope deletion.

        If scope is a customer - delete all customer estimates and their children.
        If scope is a deleted resource - redefine consumption details, recalculate
                                         ancestors estimates and update estimate details.
        If scope is a unlinked resource - delete all resource price estimates and update ancestors.
        In all other cases - update price estimate details.
    """

    is_resource = isinstance(instance, structure_models.ResourceMixin)
    if is_resource and getattr(instance, 'PERFORM_UNLINK', False):
        _resource_unlink(resource=instance)
    elif is_resource and not getattr(instance, 'PERFORM_UNLINK', False):
        _resource_deletion(resource=instance)
    elif isinstance(instance, structure_models.Customer):
        _customer_deletion(customer=instance)
    else:
        for price_estimate in models.PriceEstimate.objects.filter(scope=instance):
            price_estimate.init_details()