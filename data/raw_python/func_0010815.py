def lookup_field_help(self, field, default=None):
        """
        Looks up the help text for the passed in field.
        """
        help = None

        # is there a label specified for this field
        if field in self.field_config and 'help' in self.field_config[field]:
            help = self.field_config[field]['help']

        # if we were given a default, use that
        elif default:
            help = default

        # try to see if there is a description on our model
        elif hasattr(self, 'model'):
            for model_field in self.model._meta.fields:
                if model_field.name == field:
                    help = model_field.help_text
                    break

        return help