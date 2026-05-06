def clean(self, value):
        """Clean the provided dict of values and then return an
        EmbeddedDocument instantiated with them.
        """
        value = super(MongoEmbedded, self).clean(value)
        return self.document_class(**value)