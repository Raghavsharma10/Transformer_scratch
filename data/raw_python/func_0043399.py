def clean_prefix(self):
        """
        Validates the prefix
        """
        if self.instance.fixed:
            return self.instance.prefix

        prefix = self.cleaned_data['prefix']
        if not namespace.is_ncname(prefix):
            raise forms.ValidationError("This is an invalid prefix")

        return prefix