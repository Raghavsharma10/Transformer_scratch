def get_val(self, x):
        """Converts to int."""
        try:
            if self.subtype == 'integer':
                return int(round(x[self.col_name]))
            else:
                if np.isnan(x[self.col_name]):
                    return self.default_val

                return x[self.col_name]

        except (ValueError, TypeError):
            return self.default_val