def consistent(self,lab):
        """
        Check whether the labeling is consistent with all constraints
        """
        for const in self.constraints:
            if not const.consistent(lab):
                return False
        return True