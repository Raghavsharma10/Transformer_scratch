def _get_zone_id(self):
        """
        Will return the zone id for an existing instance.
        """
        if not self.service.exists():
            logging.warning("Service does not yet exist.")

        return self.service.settings.data['zone']['http-header-value']