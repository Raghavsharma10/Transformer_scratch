def dump(self, obj):
        """Serialize object with schema.

        :param obj: The object to serialize.
        :returns: The object serialized.
        """
        if self.schema_class:
            obj = self.schema_class().dump(obj).data
        else:
            obj = obj['metadata']
        return super(MARCXMLSerializer, self).dump(obj)