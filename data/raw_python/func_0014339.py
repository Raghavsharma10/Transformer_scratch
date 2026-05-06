def create_attributes(klass, attributes, previous_object=None):
        """
        Attributes for content type creation.
        """

        result = super(ContentType, klass).create_attributes(attributes, previous_object)

        if 'fields' not in result:
            result['fields'] = []
        return result