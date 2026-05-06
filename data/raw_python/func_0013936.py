def combine_numeric_values(self, other):
        """
        numeric_values * sp_values
        """
        if self.values is None:
            ret = IdValues()
        else:
            ret = sum([IdValues(
                {k: int(v) * int(other.values[key]) for k, v in value.items()})
                for key, value in self.values.items()])
        return ret