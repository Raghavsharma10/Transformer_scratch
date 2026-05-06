def set_pedalboard(self, pedalboard, notify=True, force=False):
        """
        Set the current :class:`.Pedalboard` for the pedalboard
        only if the ``pedalboard != current_pedalboard or force``

        Is also possible unload the current pedalboard data with replacing it for ``None``::

            >>> current_controller.set_pedalboard(None)

        .. warning::

            Changing the current pedalboard to Nonw, will not be able to call
            methods to change the pedalboard based in the current, like
            :meth:`.to_before_pedalboard`, :meth:`.to_next_pedalboard`,
            :meth:`.to_before_bank` and :meth:`.to_next_bank`

        :param Pedalboard pedalboard: New current pedalboard
        :param bool notify: If false, not notify change for :class:`.UpdatesObserver`
                            instances registered in :class:`.Application`
        :param bool force: Force set pedalboard
        """
        if pedalboard is not None and pedalboard.bank is None:
            raise CurrentPedalboardError('Pedalboard {} has not added in any bank'.format(pedalboard))

        if pedalboard == self.pedalboard and not force:
            return

        self._pedalboard = pedalboard
        self._device_controller.pedalboard = pedalboard
        self._save_current_pedalboard()

        if notify:
            self.app.components_observer.on_current_pedalboard_changed(self.pedalboard)