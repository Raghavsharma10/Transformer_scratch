def loads(self, schema_txt: str) -> ShExJ.Schema:
        """ Parse and return schema as a ShExJ Schema

        :param schema_txt: ShExC or ShExJ representation of a ShEx Schema
        :return: ShEx Schema representation of schema
        """
        self.schema_text = schema_txt
        if schema_txt.strip()[0] == '{':
            # TODO: figure out how to propagate self.base_location into this parse
            return cast(ShExJ.Schema, loads(schema_txt, ShExJ))
        else:
            return generate_shexj.parse(schema_txt, self.base_location)