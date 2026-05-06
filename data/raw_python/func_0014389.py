def create_attributes(klass, attributes, previous_object=None):
        """
        Attributes for webhook creation.
        """

        result = super(Webhook, klass).create_attributes(attributes, previous_object)

        if 'topics' not in result:
            raise Exception("Topics ('topics') must be provided for this operation.")
        return result