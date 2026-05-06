def create_attributes(klass, attributes, previous_object=None):
        """
        Attributes for resource creation.
        """

        if 'fields' not in attributes:
            if previous_object is None:
                attributes['fields'] = {}
            else:
                attributes['fields'] = previous_object.to_json()['fields']
        return {'fields': attributes['fields']}