def getContactItems(self, person):
        """
        Return a C{list} of the L{Notes} items associated with the given
        person.  If none exist, create one, wrap it in a list and return it.

        @type person: L{Person}
        """
        notes = list(person.store.query(Notes, Notes.person == person))
        if not notes:
            return [Notes(store=person.store,
                          person=person,
                          notes=u'')]
        return notes