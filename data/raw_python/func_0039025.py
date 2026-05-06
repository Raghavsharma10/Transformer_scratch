def create_manager(self, base_manager=models.Manager):
        """
        This will create the custom Manager that will use the fields_model and values_model
        respectively.

        :param base_manager: the base manager class to inherit from
        :return:
        """

        _builder = self

        class CustomManager(base_manager):
            def search(self, search_data, custom_args={}):
                """
                Search inside the custom fields for this model for any match
                 of search_data and returns existing model instances

                :param search_data:
                :param custom_args:
                :return:
                """
                query = None
                lookups = (
                    '%s__%s' % ('value_text', 'icontains'),
                )
                content_type = ContentType.objects.get_for_model(self.model)
                custom_args = dict({ 'content_type': content_type, 'searchable': True }, **custom_args)
                custom_fields = dict((f.name, f) for f in _builder.fields_model_class.objects.filter(**custom_args))
                for value_lookup in lookups:
                    for key, f in custom_fields.items():
                        found = _builder.values_model_class.objects.filter(**{ 'custom_field': f,
                                                                               'content_type': content_type,
                                                                               value_lookup: search_data })
                        if found.count() > 0:
                            if query is None:
                                query = Q()
                            query = query & Q(**{ str('%s__in' % self.model._meta.pk.name):
                                                  [obj.object_id for obj in found] })
                if query is None:
                    return self.get_queryset().none()
                return self.get_queryset().filter(query)

        return CustomManager