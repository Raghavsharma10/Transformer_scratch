def _return_rows(self, table, cols, values, return_type):
        """Return fetched rows in the desired type."""
        if return_type is dict:
            # Pack each row into a dictionary
            cols = self.get_columns(table) if cols is '*' else cols
            if len(values) > 0 and isinstance(values[0], (set, list, tuple)):
                return [dict(zip(cols, row)) for row in values]
            else:
                return dict(zip(cols, values))
        elif return_type is tuple:
            return [tuple(row) for row in values]
        else:
            return values