def groupReadOnlyViews(self, person):
        """
        Collect all contact items from the available contact types for the
        given person, organize them by contact group, and turn them into
        read-only views.

        @type person: L{Person}
        @param person: The person whose contact items we're interested in.

        @return: A mapping of of L{ContactGroup} names to the read-only views
        of their member contact items, with C{None} being the key for
        groupless contact items.
        @rtype: C{dict} of C{str}
        """
        # this is a slightly awkward, specific API, but at the time of
        # writing, read-only views are the thing that the only caller cares
        # about.  we need the contact type to get a read-only view for a
        # contact item.  there is no way to get from a contact item to a
        # contact type, so this method can't be "groupContactItems" (which
        # seems to make more sense), unless it returned some weird data
        # structure which managed to associate contact items and contact
        # types.
        grouped = {}
        for contactType in self.getContactTypes():
            for contactItem in contactType.getContactItems(person):
                contactGroup = contactType.getContactGroup(contactItem)
                if contactGroup is not None:
                    contactGroup = contactGroup.groupName
                if contactGroup not in grouped:
                    grouped[contactGroup] = []
                grouped[contactGroup].append(
                    contactType.getReadOnlyView(contactItem))
        return grouped