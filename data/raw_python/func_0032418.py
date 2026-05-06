def calculate(self, operation=None, trace=None, constant=None, type=None):
        """Starts the calculation.

        The calculation operates on the trace graphed in the active display.
        The math operation is defined by the :attr:`~.SR850.math_operation`,
        the second argument by the :attr:`~.SR850.math_argument_type`.

        For convenience, the operation and the second argument, can be
        specified via the parameters

        :param operation: Set's the math operation if not `None`. See
            :attr:`~.SR850.math_operation` for details.
        :param trace: If the trace argument is used, it sets the
            :attr:`~.math_trace_argument` to it and sets the
            :attr:`~.math_argument_type` to 'trace'
        :param constant: If constant is not `None`, the
            :attr:`~.math_constant`is set with this value and the
            :attr:`~.math_argument_type` is set to 'constant'
        :param type: If type is not `None`, the :attr:`~.math_argument_type` is
            set to this value.

        E.g. instead of::

            lockin.math_operation = '*'
            lockin.math_argument_type = 'constant'
            lockin.math_constant = 1.337
            lockin.calculate()

        one can write::

            lockin.calculate(operation='*', constant=1.337)

        .. note:: Do not use trace, constant and type together.

        .. note::

            The calculation takes some time. Check the status byte to see when
            the operation is done. A running scan will be paused until the
            operation is complete.

        .. warning::

            The SR850 will generate an error if the active display trace is not
            stored when the command is executed.

        """
        if operation is not None:
            self.math_operation = operation
        if trace is not None:
            self.math_trace_argument = trace
            type = 'trace'
        elif constant is not None:
            self.math_constant = constant
            type = 'constant'
        if type is not None:
            self.math_argument_type = type
        self._write('CALC')