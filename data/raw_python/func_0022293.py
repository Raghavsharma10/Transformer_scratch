def read(self, vals):
        """Read values.

        Args:
            vals (list): list of strings representing values

        """
        i = 0
        if len(vals[i]) == 0:
            self.comments_1 = None
        else:
            self.comments_1 = vals[i]
        i += 1