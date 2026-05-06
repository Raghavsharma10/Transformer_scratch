def getPersonPluginWidget(self, name):
        """
        Return the L{PersonPluginView} for the named person.

        @type name: C{unicode}
        @param name: A value which corresponds to the I{name} attribute of an
        extant L{Person}.

        @rtype: L{PersonPluginView}
        """
        fragment = PersonPluginView(
            self.organizer.getOrganizerPlugins(),
            self.organizer.personByName(name))
        fragment.setFragmentParent(self)
        return fragment