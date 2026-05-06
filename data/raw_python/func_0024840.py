def _get_uri(self):
        """
        Will return the uri for an existing instance.
        """
        if not self.service.exists():
            logging.warning("Service does not yet exist.")

        return self.service.settings.data['uri']