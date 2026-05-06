def from_json(cls, data, json_schema_class=None):
        """ This class overwrites the from_json method thus, making sure, that if `from_json` is called from this class instance, it will provide its JSON schema as a default one """
        json_schema = cls.json_schema if json_schema_class is None else json_schema_class()
        return super(BlockVertex, cls).from_json(data=data, json_schema_class=json_schema.__class__)