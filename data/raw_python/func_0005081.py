def optout_saved(sender, instance, **kwargs):
    """
    This is a duplicte of the view code for DRF to stop future
    internal Django implementations breaking.
    """
    if instance.identity is None:
        # look up using the address_type and address
        identities = Identity.objects.filter_by_addr(
            instance.address_type, instance.address
        )
        if identities.count() == 1:
            instance.identity = identities[0]