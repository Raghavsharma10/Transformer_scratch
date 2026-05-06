def lookup_field_class(self, field, obj=None, default=None):
        """
        Looks up any additional class we should include when rendering this field
        """
        css = ""

        # is there a class specified for this field
        if field in self.field_config and 'class' in self.field_config[field]:
            css = self.field_config[field]['class']

        # if we were given a default, use that
        elif default:
            css = default

        return css