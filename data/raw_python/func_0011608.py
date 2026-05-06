def values_list(self, *args, **kwargs):
        """Return the primary keys as a list.

        The only valid call is values_list('pk', flat=True)
        """
        flat = kwargs.pop('flat', False)
        assert flat is True
        assert len(args) == 1
        assert args[0] == self.model._meta.pk.name
        return self.pks