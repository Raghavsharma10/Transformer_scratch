def get_sum_of_quotas_as_dict(cls, scopes, quota_names=None, fields=['usage', 'limit']):
        """
        Return dictionary with sum of all scopes' quotas.

        Dictionary format:
        {
            'quota_name1': 'sum of limits for quotas with such quota_name1',
            'quota_name1_usage': 'sum of usages for quotas with such quota_name1',
            ...
        }
        All `scopes` have to be instances of the same model.
        `fields` keyword argument defines sum of which fields of quotas will present in result.
        """
        if not scopes:
            return {}

        if quota_names is None:
            quota_names = cls.get_quotas_names()

        scope_models = set([scope._meta.model for scope in scopes])
        if len(scope_models) > 1:
            raise exceptions.QuotaError(_('All scopes have to be instances of the same model.'))

        filter_kwargs = {
            'content_type': ct_models.ContentType.objects.get_for_model(scopes[0]),
            'object_id__in': [scope.id for scope in scopes],
            'name__in': quota_names
        }

        result = {}
        if 'usage' in fields:
            items = Quota.objects.filter(**filter_kwargs)\
                         .values('name').annotate(usage=Sum('usage'))
            for item in items:
                result[item['name'] + '_usage'] = item['usage']

        if 'limit' in fields:
            unlimited_quotas = Quota.objects.filter(limit=-1, **filter_kwargs)
            unlimited_quotas = list(unlimited_quotas.values_list('name', flat=True))
            for quota_name in unlimited_quotas:
                result[quota_name] = -1

            items = Quota.objects\
                         .filter(**filter_kwargs)\
                         .exclude(name__in=unlimited_quotas)\
                         .values('name')\
                         .annotate(limit=Sum('limit'))
            for item in items:
                result[item['name']] = item['limit']

        return result