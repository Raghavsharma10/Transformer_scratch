def set_resource_type(self, klass):
        """
        set type to load and load schema
        """
        self.resource_type = klass
        self.schema = loaders.load_schema_raw(self.resource_type)