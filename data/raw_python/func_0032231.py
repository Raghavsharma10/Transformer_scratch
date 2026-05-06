def contactInfo(self, request, tag):
        """
        Render the result of calling L{IContactType.getReadOnlyView} on the
        corresponding L{IContactType} for each piece of contact info
        associated with L{person}.  Arrange the result by group, using
        C{tag}'s I{contact-group} pattern. Groupless contact items will have
        their views yielded directly.

        The I{contact-group} pattern appears once for each distinct
        L{ContactGroup}, with the following slots filled:
          I{name} - The group's C{groupName}.
          I{views} - A sequence of read-only views belonging to the group.
        """
        groupPattern = inevow.IQ(tag).patternGenerator('contact-group')
        groupedViews = self.organizer.groupReadOnlyViews(self.person)
        for (groupName, views) in sorted(groupedViews.items()):
            if groupName is None:
                yield views
            else:
                yield groupPattern().fillSlots(
                    'name', groupName).fillSlots(
                    'views', views)