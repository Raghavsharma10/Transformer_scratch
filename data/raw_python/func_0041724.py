def prefetchDeclarativeIds(self, Declarative, count) -> Deferred:
        """ Get PG Sequence Generator

        A PostGreSQL sequence generator returns a chunk of IDs for the given
        declarative.

        :return: A generator that will provide the IDs
        :rtype: an iterator, yielding the numbers to assign

        """
        return self._dbConn.prefetchDeclarativeIds(Declarative=Declarative, count=count)