def customize_form_field(self, name, field):
        """
        Allows views to customize their form fields.  By default, Smartmin replaces the plain textbox
        date input with it's own DatePicker implementation.
        """
        if isinstance(field, forms.fields.DateField) and isinstance(field.widget, forms.widgets.DateInput):
            field.widget = widgets.DatePickerWidget()
            field.input_formats = [field.widget.input_format[1]] + list(field.input_formats)

        if isinstance(field, forms.fields.ImageField) and isinstance(field.widget, forms.widgets.ClearableFileInput):
            field.widget = widgets.ImageThumbnailWidget()

        return field