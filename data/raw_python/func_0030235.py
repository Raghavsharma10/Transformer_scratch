def get_widget(self):
        """
        Create the widget for the URL type.
        """
        form_field = self.get_form_field()
        widget = form_field.widget
        if isinstance(widget, type):
            widget = widget()

        # Widget instantiation needs to happen manually.
        # Auto skip if choices is not an existing attribute.
        form_field_choices = getattr(form_field, 'choices', None)
        if form_field_choices is not None:
            if hasattr(widget, 'choices'):
                widget.choices = form_field_choices
        return widget