def set_bank(self, bank, try_preserve_index=False):
        """
        Set the current :class:`Bank` for the bank
        only if the ``bank != current_bank``

        The current pedalboard will be the first pedalboard of the new current bank
        **if it contains any pedalboard**, else will be ``None``.

        .. warning::

            If the current :attr:`.pedalboard` is ``None``, a :class:`.CurrentPedalboardError` is raised.

        :param Bank bank: Bank that will be the current
        :param bool try_preserve_index: Tries to preserve the index of the current pedalboard
                                        when changing the bank. That is, if the current pedalboard is the fifth,
                                        when updating the bank, it will attempt to place the fifth pedalboard
                                        of the new bank as the current one. If it does not get
                                        (``len(bank.pedalboards) < 6``) the current pedalboard will be the
                                        first pedalboard.
        """
        if bank not in self._manager:
            raise CurrentPedalboardError('Bank {} has not added in banks manager'.format(bank))

        if self.bank == bank:
            return

        if bank.pedalboards:
            pedalboard = self._equivalent_pedalboard(bank) if try_preserve_index else bank.pedalboards[0]
            self.set_pedalboard(pedalboard)
        else:
            self.set_pedalboard(None)