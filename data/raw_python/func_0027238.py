def get_aggregator_quotas(self, quota):
        """ Fetch ancestors quotas that have the same name and are registered as aggregator quotas. """
        ancestors = quota.scope.get_quota_ancestors()
        aggregator_quotas = []
        for ancestor in ancestors:
            for ancestor_quota_field in ancestor.get_quotas_fields(field_class=AggregatorQuotaField):
                if ancestor_quota_field.get_child_quota_name() == quota.name:
                    aggregator_quotas.append(ancestor.quotas.get(name=ancestor_quota_field))
        return aggregator_quotas