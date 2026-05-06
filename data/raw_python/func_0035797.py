def to_before_pedalboard(self):
        """
        Change the current :class:`.Pedalboard` for the previous pedalboard.

        If the current pedalboard is the first in the current :class:`Bank`,
        the current pedalboard is will be the **last of the current Bank**.

        .. warning::

            If the current :attr:`.pedalboard` is ``None``, a :class:`.CurrentPedalboardError` is raised.
        """
        if self.pedalboard is None:
            raise CurrentPedalboardError('The current pedalboard is None')

        before_index = self.pedalboard.index - 1
        if before_index == -1:
            before_index = len(self.bank.pedalboards) - 1

        self.set_pedalboard(self.bank.pedalboards[before_index])