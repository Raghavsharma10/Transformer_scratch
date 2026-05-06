def create_fields(self, base_model=models.Model, base_manager=models.Manager):
        """
        This method will create a model which will hold field types defined
        at runtime for each ContentType.

        :param base_model: base model class to inherit from
        :return:
        """

        CONTENT_TYPES = self.content_types_query

        class CustomContentTypeFieldManager(base_manager):
            pass

        @python_2_unicode_compatible
        class CustomContentTypeField(base_model):
            DATATYPE_CHOICES = (
                (CUSTOM_TYPE_TEXT,     _('text')),
                (CUSTOM_TYPE_INTEGER,  _('integer')),
                (CUSTOM_TYPE_FLOAT,    _('float')),
                (CUSTOM_TYPE_TIME,     _('time')),
                (CUSTOM_TYPE_DATE,     _('date')),
                (CUSTOM_TYPE_DATETIME, _('datetime')),
                (CUSTOM_TYPE_BOOLEAN,  _('boolean')),
            )

            content_type = models.ForeignKey(ContentType,
                                             verbose_name=_('content type'),
                                             related_name='+',
                                             limit_choices_to=CONTENT_TYPES)
            name = models.CharField(_('name'), max_length=100, db_index=True)
            label = models.CharField(_('label'), max_length=100)
            data_type = models.CharField(_('data type'), max_length=8, choices=DATATYPE_CHOICES, db_index=True)
            help_text = models.CharField(_('help text'), max_length=200, blank=True, null=True)
            required = models.BooleanField(_('required'), default=False)
            searchable = models.BooleanField(_('searchable'), default=True)
            initial = models.CharField(_('initial'), max_length=200, blank=True, null=True)
            min_length = models.PositiveIntegerField(_('min length'), blank=True, null=True)
            max_length = models.PositiveIntegerField(_('max length'), blank=True, null=True)
            min_value = models.FloatField(_('min value'), blank=True, null=True)
            max_value = models.FloatField(_('max value'), blank=True, null=True)

            objects = CustomContentTypeFieldManager()

            class Meta:
                verbose_name = _('custom field')
                verbose_name_plural = _('custom fields')
                abstract = True

            def save(self, *args, **kwargs):
                super(CustomContentTypeField, self).save(*args, **kwargs)

            def clean(self):
                # if field is required must issue a initial value
                if self.required:
                    # TODO - must create values for all instances that have not
                    #print model.objects.values_list('pk', flat=True)
                    #print self.field.filter(content_type=self.content_type)
                    #objs = self.field.filter(content_type=self.content_type) \
                    #                 .exclude(object_id__in=model.objects.values_list('pk', flat=True))
                    #for obj in objs:
                    #    print obj
                    pass

            def _check_validate_already_defined_in_model(self):
                model = self.content_type.model_class()
                if self.name in [f.name for f in model._meta.fields]:
                    raise ValidationError({ 'name': (_('Custom field already defined as model field for content type %(model_name)s') % {'model_name': model.__name__},) })

            def _check_validate_already_defined_in_custom_fields(self):
                model = self.content_type.model_class()
                qs = self.__class__._default_manager.filter(
                    content_type=self.content_type,
                    name=self.name,
                )
                if not self._state.adding and self.pk is not None:
                    qs = qs.exclude(pk=self.pk)
                if qs.exists():
                    raise ValidationError({ 'name': (_('Custom field already defined for content type %(model_name)s') % {'model_name': model.__name__},) })

            def __str__(self):
                return "%s" % self.name

        return CustomContentTypeField