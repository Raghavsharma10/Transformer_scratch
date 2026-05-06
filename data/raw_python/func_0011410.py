def _pfp__snapshot(self, recurse=True):
        """Save off the current value of the field
        """
        super(Struct, self)._pfp__snapshot(recurse=recurse)

        if recurse:
            for child in self._pfp__children:
                child._pfp__snapshot(recurse=recurse)