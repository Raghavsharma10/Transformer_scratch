def combine_sp_values(self, other):
        """
        sp_values * sp_values
        """
        if self.values is None:
            ret = IdValues()
        else:
            ret = IdValues({k: int(v) * int(other.values[k]) for k, v in
                            self.values.items()})
        return ret