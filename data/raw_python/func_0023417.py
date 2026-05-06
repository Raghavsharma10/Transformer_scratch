def add_val(self, val):
        """add value in form of dict"""
        if not isinstance(val, type({})):
            raise ValueError(type({}))
        self.read()
        self.config.update(val)
        self.save()