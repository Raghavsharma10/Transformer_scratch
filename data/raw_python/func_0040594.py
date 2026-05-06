def is_discrete(self):
        """
        Check whether this interval contains exactly one number

        :rtype: bool
        """
        return self.bounds[1] == self.bounds[0] and\
               self.included == (True,True)