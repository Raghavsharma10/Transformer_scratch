def first_image(self):
        """Ready-only attribute that provides the value of the first non-none image that's
        not the thumbnail override field.
        """
        # loop through image fields and grab the first non-none one
        for model_field in self._meta.fields:
            if isinstance(model_field, ImageField):
                if model_field.name is not 'thumbnail_override':
                    field_value = getattr(self, model_field.name)
                    if field_value.id is not None:
                        return field_value

        # no non-none images, return None
        return None