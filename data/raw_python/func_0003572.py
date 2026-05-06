def receive_information_confirmation(self, message):
        """A InformationConfirmation is received.

        If :meth:`the api version is supported
        <AYABInterface.communication.Communication.api_version_is_supported>`,
        the communication object transitions into a
        :class:`InitializingMachine`, if unsupported, into a
        :class:`UnsupportedApiVersion`
        """
        if message.api_version_is_supported():
            self._next(InitializingMachine)
        else:
            self._next(UnsupportedApiVersion)

        self._communication.controller = message