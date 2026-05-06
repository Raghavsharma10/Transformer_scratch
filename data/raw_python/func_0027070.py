def handle_aggregated_quotas(sender, instance, **kwargs):
    """ Call aggregated quotas fields update methods """
    quota = instance
    # aggregation is not supported for global quotas.
    if quota.scope is None:
        return
    quota_field = quota.get_field()
    # usage aggregation should not count another usage aggregator field to avoid calls duplication.
    if isinstance(quota_field, fields.UsageAggregatorQuotaField) or quota_field is None:
        return
    signal = kwargs['signal']
    for aggregator_quota in quota_field.get_aggregator_quotas(quota):
        field = aggregator_quota.get_field()
        if signal == signals.post_save:
            field.post_child_quota_save(aggregator_quota.scope, child_quota=quota, created=kwargs.get('created'))
        elif signal == signals.pre_delete:
            field.pre_child_quota_delete(aggregator_quota.scope, child_quota=quota)