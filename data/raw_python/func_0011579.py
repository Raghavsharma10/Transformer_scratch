def choice_default_loader(self, pk):
        """Load a Choice from the database."""
        try:
            obj = Choice.objects.get(pk=pk)
        except Choice.DoesNotExist:
            return None
        else:
            self.choice_default_add_related_pks(obj)
            return obj