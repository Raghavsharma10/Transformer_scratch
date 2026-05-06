def create_item(self, **kwargs):
        """
        Return a model instance created from kwargs.
        """
        item, created = self.queryset.model.objects.get_or_create(**kwargs)
        return item