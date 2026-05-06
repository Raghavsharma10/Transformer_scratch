def create_new_values(self):
        """
        Create values created by the user input. Return the model instances QS.
        """
        model = self.queryset.model
        pks = []
        extra_create_kwargs = self.extra_create_kwargs()
        for value in self._new_values:
            create_kwargs = {self.create_field: value}
            create_kwargs.update(extra_create_kwargs)
            new_item = self.create_item(**create_kwargs)
            pks.append(new_item.pk)
        return model.objects.filter(pk__in=pks)