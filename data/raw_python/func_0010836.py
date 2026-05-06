def lookup_field_help(self, field, default=None):
        """
        Looks up the help text for the passed in field.

        This is overloaded so that we can check whether our form has help text set
        explicitely.  If so, we will pass this as the default to our parent function.
        """
        default = None

        for form_field in self.form:
            if form_field.name == field:
                default = form_field.help_text
                break

        return super(SmartFormMixin, self).lookup_field_help(field, default=default)