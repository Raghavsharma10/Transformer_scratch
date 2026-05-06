def get_field_info(self, field, field_name):
        """
        Given an instance of a serializer field, return a dictionary
        of metadata about it.
        """
        field_info = OrderedDict()
        field_info['type'] = self.label_lookup[field]
        field_info['required'] = getattr(field, 'required', False)

        attrs = [
            'label', 'help_text', 'default_value', 'placeholder', 'required',
            'min_length', 'max_length', 'min_value', 'max_value', 'many'
        ]

        if getattr(field, 'read_only', False):
            return None

        for attr in attrs:
            value = getattr(field, attr, None)
            if value is not None and value != '':
                field_info[attr] = force_text(value, strings_only=True)

        if 'label' not in field_info:
            field_info['label'] = field_name.replace('_', ' ').title()

        if hasattr(field, 'view_name'):
            list_view = field.view_name.replace('-detail', '-list')
            base_url = reverse(list_view, request=self.request)
            field_info['type'] = 'select'
            field_info['url'] = base_url
            if hasattr(field, 'query_params'):
                field_info['url'] += '?%s' % urlencode(field.query_params)
            field_info['value_field'] = getattr(field, 'value_field', 'url')
            field_info['display_name_field'] = getattr(field, 'display_name_field', 'display_name')

        if hasattr(field, 'choices') and not hasattr(field, 'queryset'):
            field_info['choices'] = [
                {
                    'value': choice_value,
                    'display_name': force_text(choice_name, strings_only=True)
                }
                for choice_value, choice_name in field.choices.items()
            ]

        return field_info