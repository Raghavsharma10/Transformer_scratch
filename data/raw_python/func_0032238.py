def getEditPerson(self, name):
        """
        Get an L{EditPersonView} for editing the person named C{name}.

        @param name: A person name.
        @type name: C{unicode}

        @rtype: L{EditPersonView}
        """
        view = EditPersonView(self.organizer.personByName(name))
        view.setFragmentParent(self)
        return view