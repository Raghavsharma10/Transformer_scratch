def lookup_field_label(self, context, field, default=None):
        """
        Figures out what the field label should be for the passed in field name.

        Our heuristic is as follows:
            1) we check to see if our field_config has a label specified
            2) if not, then we derive a field value from the field name
        """
        # if this is a subfield, strip off everything but the last field name
        if field.find('.') >= 0:
            return self.lookup_field_label(context, field.split('.')[-1], default)

        label = None

        # is there a label specified for this field
        if field in self.field_config and 'label' in self.field_config[field]:
            label = self.field_config[field]['label']

        # if we were given a default, use that
        elif default:
            label = default

        # check our model
        else:
            for model_field in self.model._meta.fields:
                if model_field.name == field:
                    return model_field.verbose_name.title()

        # otherwise, derive it from our field name
        if label is None:
            label = self.derive_field_label(field)

        return label