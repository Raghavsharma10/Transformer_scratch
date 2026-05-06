def formfield_for_dbfield(self, db_field, **kwargs):
        """
        Same as parent but sets the widget for any OrderFields to
        HiddenTextInput.
        """
        if isinstance(db_field, fields.OrderField):
            kwargs['widget'] = widgets.HiddenTextInput

        return super(ListView, self).formfield_for_dbfield(db_field, **kwargs)