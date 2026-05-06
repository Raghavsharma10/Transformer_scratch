def enter(self):
        """Send a LineConfirmation to the controller.

        When this state is entered, a
        :class:`AYABInterface.communication.host_messages.LineConfirmation`
        is sent to the controller.
        Also, the :attr:`last line requested
        <AYABInterface.communication.Communication.last_requested_line_number>`
        is set.
        """
        self._communication.last_requested_line_number = self._line_number
        self._communication.send(LineConfirmation, self._line_number)