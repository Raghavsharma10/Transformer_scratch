def prefetchDeclarativeIds(self, Declarative, count) -> DelcarativeIdGen:
        """ Prefetch Declarative IDs

        This function prefetches a chunk of IDs from a database sequence.
        Doing this allows us to preallocate the IDs before an insert, which significantly
        speeds up :

        * Orm inserts, especially those using inheritance
        * When we need the ID to assign it to a related object that we're also inserting.

        :param Declarative: The SQLAlchemy declarative class.
            (The class that inherits from DeclarativeBase)

        :param count: The number of IDs to prefetch

        :return: An iterable that dispenses the new IDs
        """
        return _commonPrefetchDeclarativeIds(
            self.dbEngine, self._sequenceMutex, Declarative, count
        )