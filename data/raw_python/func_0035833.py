def find_dimension_by_name(self, dim_name):
        """the method searching dimension with a given name"""

        for dim in self.dimensions:
            if is_equal_strings_ignore_case(dim.name, dim_name):
                return dim
        return None