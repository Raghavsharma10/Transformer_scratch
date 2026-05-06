def get_form_class(self):
        """
        Returns the form class to use in this view. Makes
        sure that the form_field_callback is set to use
        the `formfield_for_dbfield` method and that any
        custom form classes are prepared by the
        `customize_form_widgets` method.
        """
        if self.fieldsets:
            fields = flatten_fieldsets(self.get_fieldsets())
        else:
            if (self.form_class and
                    getattr(self.form_class, 'Meta', None) and
                    getattr(self.form_class.Meta, 'fields', None)):
                fields = self.form_class.Meta.fields
            else:
                fields = []

        exclude = None
        if self.parent_field:
            exclude = (self.parent_field,)

        readonly_fields = self.get_readonly_fields()
        if readonly_fields:
            if exclude:
                exclude = list(exclude)
            else:
                exclude = []

            for field in readonly_fields:
                try:
                    try:
                        f = self.model._meta.get_field(field)
                        if fields:
                            fields.remove(field)
                        else:
                            exclude.append(field)
                    except models.FieldDoesNotExist:
                        if fields:
                            fields.remove(field)
                except ValueError:
                    pass

        params = {'fields': fields or '__all__',
                  'exclude': exclude,
                  'formfield_callback': self.formfield_for_dbfield}

        if self.form_class:
            if issubclass(self.form_class, forms.ModelForm) and \
                    getattr(self.form_class._meta, 'model', None):
                model = self.form_class.Meta.model
            else:
                model = self.model
            fc = self.customize_form_widgets(self.form_class, fields=fields)
            params['form'] = fc
        else:
            if self.model is not None:
                # If a model has been explicitly provided, use it
                model = self.model
            elif hasattr(self, 'object') and self.object is not None:
                # If this view is operating on a single object, use
                # the class of that object
                model = self.object.__class__
            else:
                # Try to get a queryset and extract the model class
                # from that
                model = self.get_queryset().model

        return model_forms.modelform_factory(model, **params)