def find_dimension_by_id(self, dim_id):
        """the method searching dimension with a given id"""

        for dim in self.dimensions:
            if is_equal_strings_ignore_case(dim.id, dim_id):
                return dim
        return None