def _getSortKey(self, planet):
        """ Takes a planet and turns it into a key to be sorted by
        :param planet:
        :return:
        """

        value = eval('planet.'+self._planetProperty)

        # TODO some sort of data validation, either before or using try except

        if self.unit is not None:
            try:
                value = value.rescale(self.unit)
            except AttributeError:  # either nan or unitless
                pass

        return _sortValueIntoGroup(self._allowedKeys[:-1], self._binlimits, value)