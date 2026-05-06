def getPeopleTags(self):
        """
        Return a sequence of tags which have been applied to L{Person} items.

        @rtype: C{set}
        """
        query = self.store.query(
            Tag, Tag.object == Person.storeID)
        return set(query.getColumn('name').distinct())