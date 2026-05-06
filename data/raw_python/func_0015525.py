def load(self, schema_file: Union[str, TextIO], schema_location: Optional[str]=None) -> ShExJ.Schema:
        """ Load a ShEx Schema from schema_location

        :param schema_file:  name or file-like object to deserialize
        :param schema_location: URL or file name of schema.  Used to create the base_location
        :return: ShEx Schema represented by schema_location
        """
        if isinstance(schema_file, str):
            schema_file = self.location_rewrite(schema_file)
            self.schema_text = load_shex_file(schema_file)
        else:
            self.schema_text = schema_file.read()

        if self.base_location:
            self.root_location = self.base_location
        elif schema_location:
            self.root_location = os.path.dirname(schema_location) + '/'
        else:
            self.root_location = None
        return self.loads(self.schema_text)