def validate_template_name(self, key, value):
        """Validate template name.

        :param key: The template path.
        :param value: The template name.
        :raises ValueError: If template name is wrong.
        """
        if value not in dict(current_app.config['PAGES_TEMPLATES']):
            raise ValueError(
                'Template "{0}" does not exist.'.format(value))
        return value