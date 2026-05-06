def switch(self, name):
        """
        Returns the switch with the provided ``name``.

        If ``autocreate`` is set to ``True`` and no switch with that name
        exists, a ``DISABLED`` switch will be with that name.

        Keyword Arguments:
        name -- A name of a switch.
        """
        try:
            switch = self.storage[self.__namespaced(name)]
        except KeyError:
            if not self.autocreate:
                raise ValueError("No switch named '%s' registered in '%s'" % (name, self.namespace))

            switch = self.__create_and_register_disabled_switch(name)

        switch.manager = self
        return switch