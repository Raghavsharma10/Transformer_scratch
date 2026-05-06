def register_members(self):
        """Collect the names of the class member and convert them to object
        members.

        Unlike Terms, the Group class members are converted into object
        members, so the configuration data

        """

        self._members = {
            name: attr for name, attr in iteritems(type(self).__dict__) if isinstance(attr, Group)}

        for name, m in iteritems(self._members):
            m.init_descriptor(name, self)