def receive_start_confirmation(self, message):
        """Receive a StartConfirmation message.

        :param message: a :class:`StartConfirmation
          <AYABInterface.communication.hardware_messages.StartConfirmation>`
          message

        If the message indicates success, the communication object transitions
        into :class:`KnittingStarted` or else, into :class:`StartingFailed`.
        """
        if message.is_success():
            self._next(KnittingStarted)
        else:
            self._next(StartingFailed)