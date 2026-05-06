def customize_form_widgets(self, form_class, fields=None):
        """
        Hook for customizing widgets for a form_class. This is needed
        for forms that specify their own fields causing the
        default db_field callback to not be run for that field.

        Default implementation checks for APIModelChoiceWidgets
        or APIManyChoiceWidgets and runs the update_links method
        on them. Passing the admin_site and request being used.

        Returns a new class that contains the field with the initialized
        custom widget.
        """
        attrs = {}
        if fields:
            fields = set(fields)

        for k, f in form_class.base_fields.items():
            if fields and not k in fields:
                continue

            if isinstance(f.widget, widgets.APIModelChoiceWidget) \
                    or isinstance(f.widget, widgets.APIManyChoiceWidget):
                field = copy.deepcopy(f)
                field.widget.update_links(self.request, self.bundle.admin_site)
                attrs[k] = field

        if attrs:
            form_class = type(form_class.__name__, (form_class,), attrs)

        return form_class