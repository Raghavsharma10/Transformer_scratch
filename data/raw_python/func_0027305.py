def validate_quota_change(self, quota_deltas, raise_exception=False):
        """
        Get error messages about object and his ancestor quotas that will be exceeded if quota_delta will be added.

        raise_exception - if True QuotaExceededException will be raised if validation fails
        quota_deltas - dictionary of quotas deltas, example:
        {
            'ram': 1024,
            'storage': 2048,
            ...
        }
        Example of output:
            ['ram quota limit: 1024, requires: 2048(instance#1)', ...]

        """
        errors = []
        for name, delta in six.iteritems(quota_deltas):
            quota = self.quotas.get(name=name)
            if quota.is_exceeded(delta):
                errors.append('%s quota limit: %s, requires %s (%s)\n' % (
                    quota.name, quota.limit, quota.usage + delta, quota.scope))
        if not raise_exception:
            return errors
        else:
            if errors:
                raise exceptions.QuotaExceededException(_('One or more quotas were exceeded: %s') % ';'.join(errors))