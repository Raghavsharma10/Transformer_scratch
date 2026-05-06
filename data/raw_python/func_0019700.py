def free_symbolic(self):
        """Free symbolic data"""
        if self._symbolic is not None:
            self.funs.free_symbolic(self._symbolic)
            self._symbolic = None
            self.mtx = None