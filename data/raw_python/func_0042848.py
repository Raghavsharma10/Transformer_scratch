def leaves(self, column_name):
        """
        :param column_name:
        :return: ALL COLUMNS THAT START WITH column_name, NOT INCLUDING DEEPER NESTED COLUMNS
        """
        clean_name = unnest_path(column_name)

        if clean_name != column_name:
            clean_name = column_name
            cleaner = lambda x: x
        else:
            cleaner = unnest_path


        columns = self.columns
        # TODO: '.' IMPLIES ALL FIELDS FROM ABSOLUTE PERPECTIVE, ALL OTHERS ARE A RELATIVE PERSPECTIVE
        # TODO: HOW TO REFER TO FIELDS THAT MAY BE SHADOWED BY A RELATIVE NAME?
        for path in reversed(self.query_path) if clean_name == '.' else self.query_path:
            output = [
                c
                for c in columns
                if (
                    (c.name != "_id" or clean_name == "_id") and
                    (
                        (c.jx_type == EXISTS and column_name.endswith("." + EXISTS_TYPE)) or
                        c.jx_type not in OBJECTS or
                        (clean_name == '.' and c.cardinality == 0)
                    ) and
                    startswith_field(cleaner(relative_field(c.name, path)), clean_name)
                )
            ]
            if output:
                return set(output)
        return set()