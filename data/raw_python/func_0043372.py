def list_adresposities_by_subadres_and_huisnummer(self, subadres, huisnummer):
        '''
        List all `adresposities` for a subadres and a :class:`Huisnummer`.

        :param subadres: A string representing a certain subadres.
        :param huisnummer: The :class:`Huisnummer` for which the \
            `adresposities` are wanted. OR A huisnummer id.
        :rtype: A :class:`list` of :class:`Adrespositie`
        '''
        try:
            hid = huisnummer.id
        except AttributeError:
            hid = huisnummer
        def creator():
            res = crab_gateway_request(
                self.client, 'ListAdrespositiesBySubadres', subadres, hid
            )
            try:
                return [Adrespositie(
                    r.AdrespositieId,
                    r.HerkomstAdrespositie
                )for r in res.AdrespositieItem]
            except AttributeError:
                return []
        if self.caches['short'].is_configured:
            key = 'ListAdrespositiesBySubadres#%s%s' % (subadres, hid)
            adresposities = self.caches['short'].get_or_create(key, creator)
        else:
            adresposities = creator()
        for a in adresposities:
            a.set_gateway(self)
        return adresposities