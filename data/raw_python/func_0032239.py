def deletePerson(self, name):
        """
        Delete the person named C{name}

        @param name: A person name.
        @type name: C{unicode}
        """
        self.organizer.deletePerson(self.organizer.personByName(name))