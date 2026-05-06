def get_type_len(self):
        """Retrieve the type and length for a data record."""
        # Check types and set type/len
        self.get_sql()
        return self.type, self.len, self.len_decimal