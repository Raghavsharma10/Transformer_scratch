def enabled_for(self, inpt):
        """
        Checks to see if this switch is enabled for the provided input.

        If ``compounded``, all switch conditions must be ``True`` for the switch
        to be enabled.  Otherwise, *any* condition needs to be ``True`` for the
        switch to be enabled.

        The switch state is then checked to see if it is ``GLOBAL`` or
        ``DISABLED``.  If it is not, then the switch is ``SELECTIVE`` and each
        condition is checked.

        Keyword Arguments:
        inpt -- An instance of the ``Input`` class.
        """

        signals.switch_checked.call(self)
        signal_decorated = partial(self.__signal_and_return, inpt)

        if self.state is self.states.GLOBAL:
            return signal_decorated(True)
        elif self.state is self.states.DISABLED:
            return signal_decorated(False)

        conditions_dict = ConditionsDict.from_conditions_list(self.conditions)
        conditions = conditions_dict.get_by_input(inpt)

        if conditions:
            result = self.__enabled_func(
                cond.call(inpt)
                for cond
                in conditions
                if cond.argument(inpt).applies
            )
        else:
            result = None

        return signal_decorated(result)