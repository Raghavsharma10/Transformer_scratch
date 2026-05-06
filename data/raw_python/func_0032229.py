def urlForViewState(self, person, viewState):
        """
        Return a url for L{OrganizerFragment} which will display C{person} in
        state C{viewState}.

        @type person: L{Person}
        @type viewState: L{ORGANIZER_VIEW_STATES} constant.

        @rtype: L{url.URL}
        """
        # ideally there would be a more general mechanism for encoding state
        # like this in a url, rather than ad-hoc query arguments for each
        # fragment which needs to do it.
        organizerURL = self._webTranslator.linkTo(self.storeID)
        return url.URL(
            netloc='', scheme='',
            pathsegs=organizerURL.split('/')[1:],
            querysegs=(('initial-person', person.name),
                       ('initial-state', viewState)))