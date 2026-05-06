def init_quotas(sender, instance, created=False, **kwargs):
    """ Initialize new instances quotas """
    if not created:
        return
    for field in sender.get_quotas_fields():
        try:
            field.get_or_create_quota(scope=instance)
        except CreationConditionFailedQuotaError:
            pass