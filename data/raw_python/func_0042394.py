def verify(self):
        """
        Running all conditions in the instance variable valid_list
        Return:
            True: pass all conditions
            False: fail at more than one condition
        """
        if self not in self._queue:
            return False
        valid = True
        for check in self.valid_list:
            valid = valid & check()
        return valid