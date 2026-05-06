def getOrCreateForeignKey(self, model_class, field_name):
        """
        Return related random object to set as ForeignKey.
        """
        # Getting related object type
        # Eg: <django.db.models.fields.related.ForeignKey: test_ForeignKey>
        instance = getattr(model_class, field_name).field

        # Getting the model name by instance to find/create first id/pk.
        # Eg: <class 'django.contrib.auth.models.User'>
        related_model = instance.related_model().__class__

        # Trying to get random id from queryset.
        objects = related_model.objects.all()
        if objects.exists():
            return self.randomize(objects)

        # Returning first object from tuple `(<User: user_name>, False)`
        return related_model.objects.get_or_create(pk=1)[0]