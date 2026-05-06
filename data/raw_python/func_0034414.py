def from_json(cls, data, json_schema_class=None):
        """ JSON deserialization method that retrieves a genome instance from its json representation

        If specific json schema is provided, it is utilized, and if not, a class specific is used
        """
        schema = cls.json_schema if json_schema_class is None else json_schema_class()
        return schema.load(data)