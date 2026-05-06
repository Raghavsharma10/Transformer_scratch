def free_numeric(self):
        """Free numeric data"""
        if self._numeric is not None:
            self.funs.free_numeric(self._numeric)
            self._numeric = None
            self.free_symbolic()