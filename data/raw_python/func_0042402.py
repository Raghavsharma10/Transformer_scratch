def map_to_es(self):
        """
        RETURN A MAP FROM THE NAMESPACE TO THE es_column NAME
        """
        full_name = self.query_path
        return set_default(
            {
                relative_field(c.name, full_name): c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.jx_type not in STRUCT
            },
            {
                c.name: c.es_column
                for k, cs in self.lookup.items()
                # if startswith_field(k, full_name)
                for c in cs if c.jx_type not in STRUCT
            }
        )