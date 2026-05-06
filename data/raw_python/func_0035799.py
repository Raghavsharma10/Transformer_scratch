def to_before_bank(self):
        """
        Change the current :class:`Bank` for the before bank. If the current
        bank is the first, the current bank is will be the last bank.

        The current pedalboard will be the first pedalboard of the new current bank
        **if it contains any pedalboard**, else will be ``None``.

        .. warning::

            If the current :attr:`.pedalboard` is ``None``, a :class:`.CurrentPedalboardError` is raised.
        """
        if self.pedalboard is None:
            raise CurrentPedalboardError('The current pedalboard is None')

        before_index = self.bank.index - 1
        if before_index == -1:
            before_index = len(self._manager.banks) - 1

        self.set_bank(self._manager.banks[before_index])