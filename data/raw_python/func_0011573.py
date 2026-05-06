def user_default_add_related_pks(self, obj):
        """Add related primary keys to a User instance."""
        if not hasattr(obj, '_votes_pks'):
            obj._votes_pks = list(obj.votes.values_list('pk', flat=True))