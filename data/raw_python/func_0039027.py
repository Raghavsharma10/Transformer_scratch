def create_modelform(self, base_form=forms.ModelForm,
                         field_types=settings.CUSTOM_FIELD_TYPES,
                         widget_types=settings.CUSTOM_WIDGET_TYPES):
        """
        This creates the class that implements a ModelForm that knows about
        the custom fields

        :param base_form:
        :param field_types:
        :param widget_types:
        :return:
        """

        _builder = self

        class CustomFieldModelBaseForm(base_form):
            def __init__(self, *args, **kwargs):
                """
                Constructor
                """
                # additional form variables
                self.custom_classes = None
                self.is_custom_form = True
                self.instance = None

                # construct the form
                super(CustomFieldModelBaseForm, self).__init__(*args, **kwargs)

                # init custom fields from model in the form
                self.init_custom_fields()

            def clean(self):
                """
                Clean the form
                """
                cleaned_data = super(CustomFieldModelBaseForm, self).clean()
                return cleaned_data

            def save(self, commit=True):
                """
                Save the form
                """
                self.instance = super(CustomFieldModelBaseForm, self).save(commit=commit)
                if self.instance and commit:
                    self.instance.save()
                    self.save_custom_fields()
                return self.instance

            def init_custom_fields(self):
                """
                Populate the ``form.fields[]`` with the additional fields coming from
                the custom fields models.
                """
                content_type = self.get_content_type()
                fields = self.get_fields_for_content_type(content_type)
                for f in fields:
                    name = str(f.name)
                    initial = f.initial
                    self.fields[name] = self.get_formfield_for_field(f)
                    self.fields[name].is_custom = True
                    self.fields[name].label = f.label
                    self.fields[name].required = f.required
                    self.fields[name].widget = self.get_widget_for_field(f)
                    if self.instance and self.instance.pk:
                        value = self.search_value_for_field(f,
                                                            content_type,
                                                            self.instance.pk)
                        if len(value) > 0:
                            initial = value[0].value
                    self.fields[name].initial = self.initial[name] = initial

            def save_custom_fields(self):
                """ Perform save and validation over the custom fields """
                if not self.instance.pk:
                    raise Exception("The model instance has not been saved. Have you called instance.save() ?")

                content_type = self.get_content_type()
                fields = self.get_fields_for_content_type(content_type)
                for f in fields:
                    name = str(f.name)
                    fv = self.search_value_for_field(f,
                                                     content_type,
                                                     self.instance.pk)
                    if len(fv) > 0:
                        value = fv[0]
                        value.value = self.cleaned_data[name]
                    else:
                        value = self.create_value_for_field(f,
                                                            self.instance.pk,
                                                            self.cleaned_data[name])
                    value.save()

            def get_model(self):
                """
                Returns the actual model this ``ModelForm`` is referring to
                """
                return self._meta.model

            def get_content_type(self):
                """
                Returns the content type instance of the model this ``ModelForm`` is
                referring to
                """
                return ContentType.objects.get_for_model(self.get_model())

            def get_formfield_for_field(self, field):
                """
                Returns the defined formfield instance built from the type of the field

                :param field: custom field instance
                :return: the formfield instance
                """
                field_attrs = {
                    'label': field.label,
                    'help_text': field.help_text,
                    'required': field.required,
                }
                if field.data_type == CUSTOM_TYPE_TEXT:
                    #widget_attrs = {}
                    if field.min_length:
                        field_attrs['min_length'] = field.min_length
                    if field.max_length:
                        field_attrs['max_length'] = field.max_length
                    #    widget_attrs['maxlength'] = field.max_length
                    #field_attrs['widget'] = widgets.AdminTextInputWidget(attrs=widget_attrs)
                elif field.data_type == CUSTOM_TYPE_INTEGER:
                    if field.min_value: field_attrs['min_value'] = int(float(field.min_value))
                    if field.max_value: field_attrs['max_value'] = int(float(field.max_value))
                    #field_attrs['widget'] = spinner.IntegerSpinnerWidget(attrs=field_attrs)
                elif field.data_type == CUSTOM_TYPE_FLOAT:
                    if field.min_value: field_attrs['min_value'] = float(field.min_value)
                    if field.max_value: field_attrs['max_value'] = float(field.max_value)
                    #field_attrs['widget'] = spinner.SpinnerWidget(attrs=field_attrs)
                elif field.data_type == CUSTOM_TYPE_TIME:
                    #field_attrs['widget'] = date.TimePickerWidget()
                    pass
                elif field.data_type == CUSTOM_TYPE_DATE:
                    #field_attrs['widget'] = date.DatePickerWidget()
                    pass
                elif field.data_type == CUSTOM_TYPE_DATETIME:
                    #field_attrs['widget'] = date.DateTimePickerWidget()
                    pass
                elif field.data_type == CUSTOM_TYPE_BOOLEAN:
                    pass
                field_type = import_class(field_types[field.data_type])
                return field_type(**field_attrs)

            def get_widget_for_field(self, field, attrs={}):
                """
                Returns the defined widget type instance built from the type of the field

                :param field: custom field instance
                :param attrs: attributes of widgets
                :return: the widget instance
                """
                return import_class(widget_types[field.data_type])(**attrs)

            def get_fields_for_content_type(self, content_type):
                """
                Returns all fields for a given content type

                Example implementation:

                  return MyCustomField.objects.filter(content_type=content_type)

                :param content_type: content type to search
                :return: the custom field instances
                """

                return _builder.fields_model_class.objects.filter(content_type=content_type)

            def search_value_for_field(self, field, content_type, object_id):
                """
                This function will return the CustomFieldValue instance for a given
                field of an object that has the given content_type

                Example implementation:

                  return MyCustomFieldValue.objects.filter(custom_field=field,
                                                           content_type=content_type,
                                                           object_id=object_id)

                :param field: the custom field instance
                :param content_type: the content type instance
                :param object_id: the object id this value is referring to
                :return: CustomFieldValue queryset
                """
                return _builder.values_model_class.objects.filter(custom_field=field,
                                                                  content_type=content_type,
                                                                  object_id=object_id)

            def create_value_for_field(self, field, object_id, value):
                """
                Create a value for a given field of an object

                Example implementation:

                  return MyCustomFieldValue(custom_field=field,
                                            object_id=object_id,
                                            value=value)

                :param field: the custom field instance
                :param object_id: the object id this value is referring to
                :param value: the value to set
                :return: the value instance (not saved!)
                """
                return _builder.values_model_class(custom_field=field,
                                                   object_id=object_id,
                                                   value=value)

        return CustomFieldModelBaseForm