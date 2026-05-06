def _auto_authenticate(self):
        """
        If we are in an app context we can authenticate immediately.
        """
        client_id = predix.config.get_env_value(predix.app.Manifest, 'client_id')
        client_secret = predix.config.get_env_value(predix.app.Manifest, 'client_secret')

        if client_id and client_secret:
            logging.info("Automatically authenticated as %s" % (client_id))
            self.uaa.authenticate(client_id, client_secret)