def get_for_model(self, model):
        """
        Returns tuple (Entry instance, created) for specified
        model instance.

        :rtype: wagtailplus.wagtailrelations.models.Entry.
        """
        return self.get_or_create(
            content_type    = ContentType.objects.get_for_model(model),
            object_id       = model.pk
        )