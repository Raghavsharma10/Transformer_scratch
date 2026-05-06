def get_indexes(self, default_indexes=None):
        """Returns the list of indexes to act on based on ES_INDEXES setting

        """
        doctype = self.type.get_mapping_type_name()
        indexes = (settings.ES_INDEXES.get(doctype) or
                   settings.ES_INDEXES['default'])
        if isinstance(indexes, six.string_types):
            indexes = [indexes]
        return super(S, self).get_indexes(default_indexes=indexes)