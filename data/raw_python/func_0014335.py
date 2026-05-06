def create_attributes(klass, attributes, previous_object=None):
        """
        Attributes for resource creation.
        """

        return {
            'name': attributes.get(
                'name',
                previous_object.name if previous_object is not None else ''
            ),
            'description': attributes.get(
                'description',
                previous_object.description if previous_object is not None else ''
            ),
            'environments': attributes.get(
                'environments',
                [e.to_json() for e in previous_object.environments] if previous_object is not None else []  # Will default to master if empty
            )
        }