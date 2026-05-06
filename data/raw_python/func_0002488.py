def detect_column_renamings(self, table_differences):
        """
        Try to find columns that only changed their names.

        :type table_differences: TableDiff
        """
        rename_candidates = {}

        for added_column_name, added_column in table_differences.added_columns.items():
            for removed_column in table_differences.removed_columns.values():
                if len(self.diff_column(added_column, removed_column)) == 0:
                    if added_column.get_name() not in rename_candidates:
                        rename_candidates[added_column.get_name()] = []

                    rename_candidates[added_column.get_name()] = (removed_column, added_column, added_column_name)

        for candidate_columns in rename_candidates.values():
            if len(candidate_columns) == 1:
                removed_column, added_column, _ = candidate_columns[0]
                removed_column_name = removed_column.get_name().lower()
                added_column_name = added_column.get_name().lower()

                if removed_column_name not in table_differences.renamed_columns:
                    table_differences.renamed_columns[removed_column_name] = added_column
                    del table_differences.added_columns[added_column_name]
                    del table_differences.removed_columns[removed_column_name]