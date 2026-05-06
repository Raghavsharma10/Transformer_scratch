def to_next_bank(self):
        """
        Change the current :class:`Bank` for the next bank. If the current
        bank is the last, the current bank is will be the first bank.

        The current pedalboard will be the first pedalboard of the new current bank
        **if it contains any pedalboard**, else will be ``None``.

        .. warning::

            If the current :attr:`.pedalboard` is ``None``, a :class:`.CurrentPedalboardError` is raised.
        """
        if self.pedalboard is None:
            raise CurrentPedalboardError('The current pedalboard is None')

        next_index = self.next_bank_index(self.bank.index)

        self.set_bank(self._manager.banks[next_index])