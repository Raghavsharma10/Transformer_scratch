def _take_ownership(self):
        """Make the Python instance take ownership of the GIBaseInfo. i.e.
        unref if the python instance gets gc'ed.
        """

        if self:
            ptr = cast(self.value, GIBaseInfo)
            _UnrefFinalizer.track(self, ptr)
            self.__owns = True