def get_index(cls):
        """Gets the index for this model.

        The index for this model is specified in `settings.ES_INDEXES`
        which is a dict of mapping type -> index name.

        By default, this uses `.get_mapping_type()` to determine the
        mapping and returns the value in `settings.ES_INDEXES` for that
        or ``settings.ES_INDEXES['default']``.

        Override this to compute it differently.

        :returns: index name to use

        """
        indexes = settings.ES_INDEXES
        index = indexes.get(cls.get_mapping_type_name()) or indexes['default']
        if not (isinstance(index, six.string_types)):
            # FIXME - not sure what to do here, but we only want one
            # index and somehow this isn't one index.
            index = index[0]
        return index