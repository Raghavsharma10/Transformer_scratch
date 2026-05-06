def choice_default_add_related_pks(self, obj):
        """Add related primary keys to a Choice instance."""
        if not hasattr(obj, '_voter_pks'):
            obj._voter_pks = obj.voters.values_list('pk', flat=True)