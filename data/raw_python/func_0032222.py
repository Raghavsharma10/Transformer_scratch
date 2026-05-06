def createPerson(self, nickname, vip=_NO_VIP):
        """
        Create a new L{Person} with the given name in this organizer.

        @type nickname: C{unicode}
        @param nickname: The value for the new person's C{name} attribute.

        @type vip: C{bool}
        @param vip: Value to set the created person's C{vip} attribute to
        (deprecated).

        @rtype: L{Person}
        """
        for person in (self.store.query(
                Person, attributes.AND(
                    Person.name == nickname,
                    Person.organizer == self))):
            raise ValueError("Person with name %r exists already." % (nickname,))
        person = Person(
            store=self.store,
            created=extime.Time(),
            organizer=self,
            name=nickname)

        if vip is not self._NO_VIP:
            warn(
                "Usage of Organizer.createPerson's 'vip' parameter"
                " is deprecated",
                category=DeprecationWarning)
            person.vip = vip

        self._callOnOrganizerPlugins('personCreated', person)
        return person