def create_mixin(self):
        """
        This will create the custom Model Mixin to attach to your custom field
        enabled model.

        :return:
        """

        _builder = self

        class CustomModelMixin(object):
            @cached_property
            def _content_type(self):
                return ContentType.objects.get_for_model(self)

            @classmethod
            def get_model_custom_fields(cls):
                """ Return a list of custom fields for this model, callable at model level """
                return _builder.fields_model_class.objects.filter(content_type=ContentType.objects.get_for_model(cls))

            def get_custom_fields(self):
                """ Return a list of custom fields for this model """
                return _builder.fields_model_class.objects.filter(content_type=self._content_type)

            def get_custom_value(self, field):
                """ Get a value for a specified custom field """
                return _builder.values_model_class.objects.get(custom_field=field,
                                                               content_type=self._content_type,
                                                               object_id=self.pk)

            def set_custom_value(self, field, value):
                """ Set a value for a specified custom field """
                custom_value, created = \
                    _builder.values_model_class.objects.get_or_create(custom_field=field,
                                                                      content_type=self._content_type,
                                                                      object_id=self.pk)
                custom_value.value = value
                custom_value.full_clean()
                custom_value.save()
                return custom_value

            #def __getattr__(self, name):
            #    """ Get a value for a specified custom field """
            #    try:
            #        obj = _builder.values_model_class.objects.get(custom_field__name=name,
            #                                                      content_type=self._content_type,
            #                                                      object_id=self.pk)
            #        return obj.value
            #    except ObjectDoesNotExist:
            #        pass
            #    return super(CustomModelMixin, self).__getattr__(name)

        return CustomModelMixin