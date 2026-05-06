def create_values(self, base_model=models.Model, base_manager=models.Manager):
        """
        This method will create a model which will hold field values for
        field types of custom_field_model.

        :param base_model:
        :param base_manager:
        :return:
        """

        _builder = self

        class CustomContentTypeFieldValueManager(base_manager):
            def create(self, **kwargs):
                """
                Subclass create in order to be able to use "value" in kwargs
                instead of using "value_%s" passing also type directly
                """
                if 'value' in kwargs:
                    value = kwargs.pop('value')
                    created_object = super(CustomContentTypeFieldValueManager, self).create(**kwargs)
                    created_object.value = value
                    return created_object
                else:
                    return super(CustomContentTypeFieldValueManager, self).create(**kwargs)

        @python_2_unicode_compatible
        class CustomContentTypeFieldValue(base_model):
            custom_field = models.ForeignKey('.'.join(_builder.fields_model),
                                             verbose_name=_('custom field'),
                                             related_name='+')
            content_type = models.ForeignKey(ContentType, editable=False,
                                             verbose_name=_('content type'),
                                             limit_choices_to=_builder.content_types_query)
            object_id = models.PositiveIntegerField(_('object id'), db_index=True)
            content_object = GenericForeignKey('content_type', 'object_id')

            value_text = models.TextField(blank=True, null=True)
            value_integer = models.IntegerField(blank=True, null=True)
            value_float = models.FloatField(blank=True, null=True)
            value_time = models.TimeField(blank=True, null=True)
            value_date = models.DateField(blank=True, null=True)
            value_datetime = models.DateTimeField(blank=True, null=True)
            value_boolean = models.NullBooleanField(blank=True)

            objects = CustomContentTypeFieldValueManager()

            def _get_value(self):
                return getattr(self, 'value_%s' % self.custom_field.data_type)

            def _set_value(self, new_value):
                setattr(self, 'value_%s' % self.custom_field.data_type, new_value)

            value = property(_get_value, _set_value)

            class Meta:
                unique_together = ('custom_field', 'content_type', 'object_id')
                verbose_name = _('custom field value')
                verbose_name_plural = _('custom field values')
                abstract = True

            def save(self, *args, **kwargs):
                # save content type as user shouldn't be able to change it
                self.content_type = self.custom_field.content_type
                super(CustomContentTypeFieldValue, self).save(*args, **kwargs)

            def validate_unique(self, exclude=None):
                qs = self.__class__._default_manager.filter(
                    custom_field=self.custom_field,
                    content_type=self.custom_field.content_type,
                    object_id=self.object_id,
                )
                if not self._state.adding and self.pk is not None:
                    qs = qs.exclude(pk=self.pk)
                if qs.exists():
                    raise ValidationError({ NON_FIELD_ERRORS: (_('A value for this custom field already exists'),) })

            def __str__(self):
                return "%s: %s" % (self.custom_field.name, self.value)

        return CustomContentTypeFieldValue