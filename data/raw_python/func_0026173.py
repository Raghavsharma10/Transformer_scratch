def clear(self):
        """
        A version of clear that also affects scalars/sections
        Also clears comments and configspec.

        Leaves other attributes alone :
            depth/main/parent are not affected
        """
        dict.clear(self)
        self.scalars = []
        self.sections = []
        self.comments = {}
        self.inline_comments = {}
        self.configspec = None
        self.defaults = []
        self.extra_values = []