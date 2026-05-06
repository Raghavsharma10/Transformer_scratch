def kind(self):
        '''The kind of this execution context.'''
        with self._mutex:
            kind = self._obj.get_kind()
            if kind == RTC.PERIODIC:
                return self.PERIODIC
            elif kind == RTC.EVENT_DRIVEN:
                return self.EVENT_DRIVEN
            else:
                return self.OTHER