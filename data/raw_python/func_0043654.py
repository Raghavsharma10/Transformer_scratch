def set_zap_authenticator(self, zap_authenticator):
        """
        Setup a ZAP authenticator.

        :param zap_authenticator: A ZAP authenticator instance to use. The
            context takes ownership of the specified instance. It will close it
            automatically when it stops. If `None` is specified, any previously
            owner instance is disowned and returned. It becomes the caller's
            responsibility to close it.
        :returns: The previous ZAP authenticator instance.
        """
        result = self._zap_authenticator

        if result:
            self.unregister_child(result)

        self._zap_authenticator = zap_authenticator

        if self.zap_client:
            self.zap_client.close()

        if self._zap_authenticator:
            self.register_child(zap_authenticator)
            self.zap_client = ZAPClient(context=self)
            self.register_child(self.zap_client)
        else:
            self.zap_client = None

        return result