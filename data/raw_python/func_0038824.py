def save(self, **kwargs):
        """
        Save and return a list of object instances.
        """
        validated_data = [
            dict(list(attrs.items()) + list(kwargs.items()))
            for attrs in self.validated_data
        ]

        if "id" in validated_data:
            ModelClass = self.Meta.model

            try:
                self.instance = ModelClass.objects.get(id=validated_data["id"])
            except ModelClass.DoesNotExist:
                pass

        return super(VideohubVideoSerializer, self).save(**kwargs)