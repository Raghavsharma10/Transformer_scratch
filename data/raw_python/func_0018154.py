def get_field_label(self, field_name, field=None):
        """ Return a label to display for a field """
        label = None
        if field is not None:
            label = getattr(field, 'verbose_name', None)
            if label is None:
                label = getattr(field, 'name', None)
        if label is None:
            label = field_name
        return label.capitalize()