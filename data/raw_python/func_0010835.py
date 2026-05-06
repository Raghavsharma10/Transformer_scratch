def lookup_field_label(self, context, field, default=None):
        """
        Figures out what the field label should be for the passed in field name.

        We overload this so as to use our form to see if there is label set there.  If so
        then we'll pass that as the default instead of having our parent derive
        the field from the name.
        """
        default = None

        for form_field in self.form:
            if form_field.name == field:
                default = form_field.label
                break

        return super(SmartFormMixin, self).lookup_field_label(context, field, default=default)