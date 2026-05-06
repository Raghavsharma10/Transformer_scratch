def agree_to_tos(self, regr):
        """
        Accept the terms-of-service for a registration.

        :param ~acme.messages.RegistrationResource regr: The registration to
            update.

        :return: The updated registration resource.
        :rtype: Deferred[`~acme.messages.RegistrationResource`]
        """
        return self.update_registration(
            regr.update(
                body=regr.body.update(
                    agreement=regr.terms_of_service)))