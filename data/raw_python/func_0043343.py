def get_gewest_by_id(self, id):
        '''
        Get a `gewest` by id.

        :param integer id: The id of a `gewest`.
        :rtype: A :class:`Gewest`.
        '''
        def creator():
            nl = crab_gateway_request(
                self.client, 'GetGewestByGewestIdAndTaalCode', id, 'nl'
            )
            fr = crab_gateway_request(
                self.client, 'GetGewestByGewestIdAndTaalCode', id, 'fr'
            )
            de = crab_gateway_request(
                self.client, 'GetGewestByGewestIdAndTaalCode', id, 'de'
            )
            if nl == None:
                raise GatewayResourceNotFoundException()
            return Gewest(
                nl.GewestId,
                {
                    'nl': nl.GewestNaam,
                    'fr': fr.GewestNaam,
                    'de': de.GewestNaam
                },
                (nl.CenterX, nl.CenterY),
                (nl.MinimumX, nl.MinimumY, nl.MaximumX, nl.MaximumY),
            )
        if self.caches['permanent'].is_configured:
            key = 'GetGewestByGewestId#%s' % id
            gewest = self.caches['long'].get_or_create(key, creator)
        else:
            gewest = creator()
        gewest.set_gateway(self)
        return gewest