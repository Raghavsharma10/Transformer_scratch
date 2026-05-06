def copy(self, schema_only=False):
        """
        Creates a deepcopy of the instance.
        If schema_only is True, the data will be excluded from the copy.
        """
        o = type(self)()
        o.relation = self.relation
        o.attributes = list(self.attributes)
        o.attribute_types = self.attribute_types.copy()
        o.attribute_data = self.attribute_data.copy()
        if not schema_only:
            o.comment = list(self.comment)
            o.data = copy.deepcopy(self.data)
        return o