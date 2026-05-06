def modify_kls(self, name):
        """Add a part to what will end up being the kls' superclass"""
        if self.kls is None:
            self.kls = name
        else:
            self.kls += name