def update_validity(self):
        """
        Update validity of a service.
        """

        # WM is always valid
        if self.type == 'Hypermap:WorldMap':
            return

        signals.post_save.disconnect(service_post_save, sender=Service)

        try:

            # some service now must be considered invalid:
            # 0. any service not exposed in SUPPORTED_SRS
            # 1. any WMTS service
            # 2. all of the NOAA layers

            is_valid = True

            # 0. any service not exposed in SUPPORTED_SRS
            if self.srs.filter(code__in=SUPPORTED_SRS).count() == 0:
                LOGGER.debug('Service with id %s is marked invalid because in not exposed in SUPPORTED_SRS' % self.id)
                is_valid = False

            # 1. any WMTS service
            if self.type == 'OGC:WMTS':
                LOGGER.debug('Service with id %s is marked invalid because it is of type OGC:WMTS' % self.id)
                is_valid = False

            # 2. all of the NOAA layers
            if 'noaa' in self.url.lower():
                LOGGER.debug('Service with id %s is marked invalid because it is from NOAA' % self.id)
                is_valid = False

            # now we save the service
            self.is_valid = is_valid
            self.save()

        except:
            LOGGER.error('Error updating validity of the service!')

        signals.post_save.connect(service_post_save, sender=Service)