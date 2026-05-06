def next(self):
        """A `next` that caches the returned results.  Together with the
        slightly different `__iter__`, these cursors can be iterated over
        more than once."""
        if self.__tailable:
            return PymongoCursor.next(self)
        try:
            ret = PymongoCursor.next(self)
        except StopIteration:
            self.__fullcache = True
            raise
        self.__itercache.append(ret)
        return ret