def register(self, new_reg=None):
        """
        Create a new registration with the ACME server.

        :param ~acme.messages.NewRegistration new_reg: The registration message
            to use, or ``None`` to construct one.

        :return: The registration resource.
        :rtype: Deferred[`~acme.messages.RegistrationResource`]
        """
        if new_reg is None:
            new_reg = messages.NewRegistration()
        action = LOG_ACME_REGISTER(registration=new_reg)
        with action.context():
            return (
                DeferredContext(
                    self.update_registration(
                        new_reg, uri=self.directory[new_reg]))
                .addErrback(self._maybe_registered, new_reg)
                .addCallback(
                    tap(lambda r: action.add_success_fields(registration=r)))
                .addActionFinish())