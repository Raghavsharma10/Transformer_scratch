def add_metadata_query_properties(self, meta_constraints, id_table, id_column):
        """
        Construct WHERE clauses from a list of MetaConstraint objects, adding them to the query state.

        :param meta_constraints:
            A list of MetaConstraint objects, each of which defines a condition over metadata which must be satisfied
            for results to be included in the overall query.
        :raises:
            ValueError if an unknown meta constraint type is encountered.
        """
        for mc in meta_constraints:
            meta_key = str(mc.key)
            ct = mc.constraint_type
            sql_template = """
{0}.uid IN (
SELECT m.{1} FROM archive_metadata m
INNER JOIN archive_metadataFields k ON m.fieldId=k.uid
WHERE m.{2} {3} %s AND k.metaKey = %s
)"""
            # Add metadata value to list of SQL arguments
            self.sql_args.append(SQLBuilder.map_value(mc.value))
            # Add metadata key to list of SQL arguments
            self.sql_args.append(meta_key)
            # Put an appropriate WHERE clause
            if ct == 'less':
                self.where_clauses.append(sql_template.format(id_table, id_column, 'floatValue', '<='))
            elif ct == 'greater':
                self.where_clauses.append(sql_template.format(id_table, id_column, 'floatValue', '>='))
            elif ct == 'number_equals':
                self.where_clauses.append(sql_template.format(id_table, id_column, 'floatValue', '='))
            elif ct == 'string_equals':
                self.where_clauses.append(sql_template.format(id_table, id_column, 'stringValue', '='))
            else:
                raise ValueError("Unknown meta constraint type!")