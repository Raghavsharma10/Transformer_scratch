def get_schema_specs_of_type(self, *schema_types: Type) -> Dict[str, Dict[str, Any]]:
        """
        Returns a list of fully qualified names and schema dictionary tuples for
        the schema types provided.
        :param schema_types: Schema types.
        :return: List of fully qualified names and schema dictionary tuples.
        """

        return {
            fq_name: schema
            for fq_name, schema in self._spec_cache.items()
            if Type.is_type_in(schema.get(ATTRIBUTE_TYPE, ''), list(schema_types))
        }