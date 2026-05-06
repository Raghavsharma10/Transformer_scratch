def match(self, attr, val):
        """ lookup object in directory with attribute matching value """
        self._lock.acquire()
        try:
            for x in self:
                if getattr(x, attr) == val:
                    return x
        finally:
            self._lock.release()