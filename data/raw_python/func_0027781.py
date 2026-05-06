def getItemByID(self, storeID, default=_noItem, autoUpgrade=True):
        """
        Retrieve an item by its storeID, and return it.

        Note: most of the failure modes of this method are catastrophic and
        should not be handled by application code.  The only one that
        application programmers should be concerned with is KeyError.  They are
        listed for educational purposes.

        @param storeID: an L{int} which refers to the store.

        @param default: if passed, return this value rather than raising in the
        case where no Item is found.

        @raise TypeError: if storeID is not an integer.

        @raise UnknownItemType: if the storeID refers to an item row in the
        database, but the corresponding type information is not available to
        Python.

        @raise RuntimeError: if the found item's class version is higher than
        the current application is aware of.  (In other words, if you have
        upgraded a database to a new schema and then attempt to open it with a
        previous version of the code.)

        @raise errors.ItemNotFound: if no item existed with the given storeID.

        @return: an Item, or the given default, if it was passed and no row
        corresponding to the given storeID can be located in the database.
        """

        if not isinstance(storeID, (int, long)):
            raise TypeError("storeID *must* be an int or long, not %r" % (
                    type(storeID).__name__,))
        if storeID == STORE_SELF_ID:
            return self
        try:
            return self.objectCache.get(storeID)
        except KeyError:
            pass
        log.msg(interface=iaxiom.IStatEvent, stat_cache_misses=1, key=storeID)
        results = self.querySchemaSQL(_schema.TYPEOF_QUERY, [storeID])
        assert (len(results) in [1, 0]),\
            "Database panic: more than one result for TYPEOF!"
        if results:
            typename, module, version = results[0]
            useMostRecent = False
            moreRecentAvailable = False

            # The schema may have changed since the last time I saw the
            # database.  Let's look to see if this is suspiciously broken...

            if _typeIsTotallyUnknown(typename, version):
                # Another process may have created it - let's re-up the schema
                # and see what we get.
                self._startup()

                # OK, all the modules have been loaded now, everything
                # verified.
                if _typeIsTotallyUnknown(typename, version):

                    # If there is STILL no inkling of it anywhere, we are
                    # almost certainly boned.  Let's tell the user in a
                    # structured way, at least.
                    raise errors.UnknownItemType(
                        "cannot load unknown schema/version pair: %r %r - id: %r" %
                        (typename, version, storeID))

            if typename in _typeNameToMostRecentClass:
                moreRecentAvailable = True
                mostRecent = _typeNameToMostRecentClass[typename]

                if mostRecent.schemaVersion < version:
                    raise RuntimeError("%s:%d - was found in the database and most recent %s is %d" %
                                       (typename, version, typename, mostRecent.schemaVersion))
                if mostRecent.schemaVersion == version:
                    useMostRecent = True
            if useMostRecent:
                T = mostRecent
            else:
                T = self.getOldVersionOf(typename, version)

            # for the moment we're going to assume no inheritance
            attrs = self.querySQL(T._baseSelectSQL(self), [storeID])
            if len(attrs) == 0:
                if default is _noItem:
                    raise errors.ItemNotFound(
                        'No results for known-to-be-good object')
                return default
            elif len(attrs) > 1:
                raise errors.DataIntegrityError(
                    'Too many results for {:d}'.format(storeID))
            attrs = attrs[0]
            x = T.existingInStore(self, storeID, attrs)
            if moreRecentAvailable and (not useMostRecent) and autoUpgrade:
                # upgradeVersion will do caching as necessary, we don't have to
                # cache here.  (It must, so that app code can safely call
                # upgradeVersion and get a consistent object out of it.)
                x = self.transact(self._upgradeManager.upgradeItem, x)
            elif not x.__legacy__:
                # We loaded the most recent version of an object
                self.objectCache.cache(storeID, x)
            return x
        if default is _noItem:
            raise errors.ItemNotFound(storeID)
        return default