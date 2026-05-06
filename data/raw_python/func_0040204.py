def satisfied(self,lab):
        """
        Check whether the labeling satisfies all constraints
        """
        for const in self.constraints:
            if not const.satisfied(lab):
                return False
        return True