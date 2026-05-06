def get_deelgemeente_by_id(self, id):
        '''
        Retrieve a `deelgemeente` by the id.

        :param string id: The id of the deelgemeente.
        :rtype: :class:`Deelgemeente`
        '''
        def creator():
            if id in self.deelgemeenten:
                dg = self.deelgemeenten[id]
                return Deelgemeente(dg['id'], dg['naam'], dg['gemeente_niscode'])
            else:
                return None

        if self.caches['permanent'].is_configured:
            key = 'GetDeelgemeenteByDeelgemeenteId#%s' % id
            deelgemeente = self.caches['permanent'].get_or_create(key, creator)
        else:
            deelgemeente = creator()
        if deelgemeente == None:
            raise GatewayResourceNotFoundException()
        deelgemeente.set_gateway(self)
        return deelgemeente