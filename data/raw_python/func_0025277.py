def set_control_output(self, name: str, value: float, *, options: dict=None) -> None:
        """Set the value of a control asynchronously.

        :param name: The name of the control (string).
        :param value: The control value (float).
        :param options: A dict of custom options to pass to the instrument for setting the value.

        Options are:
            value_type: local, delta, output. output is default.
            confirm, confirm_tolerance_factor, confirm_timeout: confirm value gets set.
            inform: True to keep dependent control outputs constant by adjusting their internal values. False is
            default.

        Default value of confirm is False.

        Default confirm_tolerance_factor is 1.0. A value of 1.0 is the nominal tolerance for that control. Passing a
        higher tolerance factor (for example 1.5) will increase the permitted error margin and passing lower tolerance
        factor (for example 0.5) will decrease the permitted error margin and consequently make a timeout more likely.
        The tolerance factor value 0.0 is a special value which removes all checking and only waits for any change at
        all and then returns.

        Default confirm_timeout is 16.0 (seconds).

        Raises exception if control with name doesn't exist.

        Raises TimeoutException if confirm is True and timeout occurs.

        .. versionadded:: 1.0

        Scriptable: Yes
        """
        self.__instrument.set_control_output(name, value, options)