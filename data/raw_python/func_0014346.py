def create_attributes(klass, attributes, previous_object=None):
        """Attributes for space creation."""

        if previous_object is not None:
            return {'name': attributes.get('name', previous_object.name)}
        return {
            'name': attributes.get('name', ''),
            'defaultLocale': attributes['default_locale']
        }