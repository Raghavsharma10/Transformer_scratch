def to_next_pedalboard(self):
        """
        Change the current :class:`.Pedalboard` for the next pedalboard.

        If the current pedalboard is the last in the current :class:`.Bank`,
        the current pedalboard is will be the **first of the current Bank**

        .. warning::

            If the current :attr:`.pedalboard` is ``None``, a :class:`.CurrentPedalboardError` is raised.
        """
        if self.pedalboard is None:
            raise CurrentPedalboardError('The current pedalboard is None')

        next_index = self.pedalboard.index + 1
        if next_index == len(self.bank.pedalboards):
            next_index = 0

        self.set_pedalboard(self.bank.pedalboards[next_index])