def sum_combined_sp_values(self, other):
        """
        sum(sp_values * sp_values)
        """
        if self.values is None:
            ret = IdValues()
        else:
            ret = IdValues({'0': sum(int(x) for x in
                                     {k: int(v) * int(other.values[k]) for k, v
                                      in self.values.items()}.values())})
        return ret