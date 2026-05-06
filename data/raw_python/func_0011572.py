def user_default_loader(self, pk):
        """Load a User from the database."""
        try:
            obj = User.objects.get(pk=pk)
        except User.DoesNotExist:
            return None
        else:
            self.user_default_add_related_pks(obj)
            return obj