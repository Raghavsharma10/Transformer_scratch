def get_foreign_keys_in_altered_table(self, diff):
        """
        :param diff: The table diff
        :type diff: eloquent.dbal.table_diff.TableDiff

        :rtype: list
        """
        foreign_keys = diff.from_table.get_foreign_keys()
        column_names = self.get_column_names_in_altered_table(diff)

        for key, constraint in foreign_keys.items():
            changed = False
            local_columns = []
            for column_name in constraint.get_local_columns():
                normalized_column_name = column_name.lower()
                if normalized_column_name not in column_names:
                    del foreign_keys[key]
                    break
                else:
                    local_columns.append(column_names[normalized_column_name])
                    if column_name != column_names[normalized_column_name]:
                        changed = True

            if changed:
                pass

        return foreign_keys