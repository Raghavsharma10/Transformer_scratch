def diff_table(self, table1, table2):
        """
        Returns the difference between the tables table1 and table2.

        :type table1: Table
        :type table2: Table

        :rtype: TableDiff
        """
        changes = 0
        table_differences = TableDiff(table1.get_name())
        table_differences.from_table = table1

        table1_columns = table1.get_columns()
        table2_columns = table2.get_columns()

        # See if all the fields in table1 exist in table2
        for column_name, column in table2_columns.items():
            if not table1.has_column(column_name):
                table_differences.added_columns[column_name] = column
                changes += 1

        # See if there are any removed fields in table2
        for column_name, column in table1_columns.items():
            if not table2.has_column(column_name):
                table_differences.removed_columns[column_name] = column
                changes += 1
                continue

            # See if column has changed properties in table2
            changed_properties = self.diff_column(column, table2.get_column(column_name))

            if changed_properties:
                column_diff = ColumnDiff(column.get_name(),
                                         table2.get_column(column_name),
                                         changed_properties)
                column_diff.from_column = column
                table_differences.changed_columns[column.get_name()] = column_diff
                changes += 1

        self.detect_column_renamings(table_differences)

        # table1_indexes = table1.get_indexes()
        # table2_indexes = table2.get_indexes()
        #
        # # See if all the fields in table1 exist in table2
        # for index_name, index in table2_indexes.items():
        #     if (index.is_primary() and table1.has_primary_key()) or table1.has_index(index_name):
        #         continue
        #
        #     table_differences.added_indexes[index_name] = index
        #     changes += 1
        #
        # # See if there are any removed fields in table2
        # for index_name, index in table1_indexes.items():
        #     if (index.is_primary() and not table2.has_primary_key())\
        #             or (not index.is_primary() and not table2.has_index(index_name)):
        #         table_differences.removed_indexes[index_name] = index
        #         changes += 1
        #         continue
        #
        #     if index.is_primary():
        #         table2_index = table2.get_primary_key()
        #     else:
        #         table2_index = table2.get_index(index_name)
        #
        #     if self.diff_index(index, table2_index):
        #         table_differences.changed_indexes[index_name] = index
        #         changes += 1
        #
        # self.detect_index_renamings(table_differences)
        #
        # from_fkeys = table1.get_foreign_keys()
        # to_fkeys = table2.get_foreign_keys()
        #
        # for key1, constraint1 in from_fkeys.items():
        #     for key2, constraint2 in to_fkeys.items():
        #         if self.diff_foreign_key(constraint1, constraint2) is False:
        #             del from_fkeys[key1]
        #             del to_fkeys[key2]
        #         else:
        #             if constraint1.get_name().lower() == constraint2.get_name().lower():
        #                 table_differences.changed_foreign_keys.append(constraint2)
        #                 changes += 1
        #                 del from_fkeys[key1]
        #                 del to_fkeys[key2]
        #
        # for constraint1 in from_fkeys.values():
        #     table_differences.removed_foreign_keys.append(constraint1)
        #     changes += 1
        #
        # for constraint2 in to_fkeys.values():
        #     table_differences.added_foreign_keys.append(constraint2)
        #     changes += 1

        if changes:
            return table_differences

        return False