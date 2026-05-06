def group_default_invalidator(self, obj):
        """Invalidated cached items when the Group changes."""
        user_pks = User.objects.values_list('pk', flat=True)
        return [('User', pk, False) for pk in user_pks]