def add_star_prefix(self):
        """
        Adds a /* prefix if there is no prefix.  This is when you need
        to keep context's constrained to a single parent.
        """
        if self.path:
            self.path += '*/'
        else:
            self.path = '*/'
        self.star_prefix = True