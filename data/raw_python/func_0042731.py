def _add(self, column):
        """
        :param column: ANY COLUMN OBJECT
        :return:  None IF column IS canonical ALREADY (NET-ZERO EFFECT)
        """
        columns_for_table = self.data.setdefault(column.es_index, {})
        existing_columns = columns_for_table.setdefault(column.name, [])

        for canonical in existing_columns:
            if canonical is column:
                return None
            if canonical.es_type == column.es_type:
                if column.last_updated > canonical.last_updated:
                    for key in Column.__slots__:
                        old_value = canonical[key]
                        new_value = column[key]
                        if new_value == None:
                            pass  # DO NOT BOTHER CLEARING OLD VALUES (LIKE cardinality AND paritiions)
                        elif new_value == old_value:
                            pass  # NO NEED TO UPDATE WHEN NO CHANGE MADE (COMMON CASE)
                        else:
                            canonical[key] = new_value
                return canonical
        existing_columns.append(column)
        return column