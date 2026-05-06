def map_to_es(self):
        """
        RETURN A MAP FROM THE NAMESPACE TO THE es_column NAME
        """
        output = {}
        for path in self.query_path:
            set_default(
                output,
                {
                    k: c.es_column
                    for c in self.columns
                    if c.jx_type not in STRUCT
                    for rel_name in [relative_field(c.name, path)]
                    for k in [rel_name, untype_path(rel_name), unnest_path(rel_name)]
                }
            )
        return output