def validate_row(self, row):
        """
        Ensure each element in the row matches the schema.
        """
        clean_row = {}
        if isinstance(row, (tuple, list)):
            assert self.header_order, "No attribute order specified."
            assert len(row) == len(self.header_order), \
                "Row length does not match header length."
            itr = zip(self.header_order, row)
        else:
            assert isinstance(row, dict)
            itr = iteritems(row)
        for el_name, el_value in itr:
            if self.header_types[el_name] == ATTR_TYPE_DISCRETE:
                clean_row[el_name] = int(el_value)
            elif self.header_types[el_name] == ATTR_TYPE_CONTINUOUS:
                clean_row[el_name] = float(el_value)
            else:
                clean_row[el_name] = el_value
        return clean_row