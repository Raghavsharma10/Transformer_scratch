def applyFilters(self, headers, table):
        """Apply filter on ps command result.
        
        @param headers: List of column headers.
        @param table:   Nested list of rows and columns.
        @return:        Nested list of rows and columns filtered using 
                        registered filters.
                        
        """
        result = []
        column_idxs = {}
        for column in self._filters.keys():
            try:
                column_idxs[column] = headers.index(column)
            except ValueError:
                raise ValueError('Invalid column name %s in filter.' % column)
        for row in table:
            for (column, (patterns, 
                          is_regex, 
                          ignore_case)) in self._filters.items():
                col_idx = column_idxs[column]
                col_val = row[col_idx]
                if is_regex:
                    for pattern in patterns:
                        if pattern.search(col_val):
                            break
                    else:
                        break
                else:
                    if ignore_case:
                        col_val = col_val.lower()
                    if col_val in patterns:
                        pass
                    else:
                        break
            else:
                result.append(row)
        return result