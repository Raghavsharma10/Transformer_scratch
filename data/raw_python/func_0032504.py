def clone(self, choices):
        """
        Make a copy of this parameter, supply different choices.

        @param choices: A sequence of L{Option} instances.
        @type choices: C{list}

        @rtype: L{ChoiceParameter}
        """
        return self.__class__(
            self.name,
            choices,
            self.label,
            self.description,
            self.multiple,
            self.viewFactory)